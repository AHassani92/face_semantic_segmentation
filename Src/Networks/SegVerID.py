# torch utilities
import math
import torch
import torchvision
import torch.nn as nn
from torchmetrics import Accuracy
import segmentation_models_pytorch as smp
from torch.nn import functional as F
import numpy as np

# repository helper functions
from Src.Utils.Statistics import sample_valid_labels

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class ArcMarginProduct(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, num_embeddings = 512, num_classes = 51332, scale_factor=30., margin=0.5):
        super(ArcMarginProduct, self).__init__()
        self.num_classes = num_classes
        self.kernel = nn.Parameter(torch.Tensor(num_embeddings, num_classes))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin  # the margin value, default is 0.5
        self.scale_factor = scale_factor  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.mm = self.sin_margin * margin  # issue 1

    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)

        # calculate the dot product
        cos_theta = torch.mm(embbedings, kernel_norm)

        # for numerical stability
        cos_theta = cos_theta.clamp(-1, 1)

        # euler's formula
        sin_theta = torch.sqrt(1 - torch.pow(cos_theta, 2))

        # angular margin
        phi = (cos_theta * self.cos_margin - sin_theta * self.sin_margin)

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m

        cond_v = cos_theta - self.theta
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        phi[cond_mask] = keep_val[cond_mask]
        logits = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        logits[idx_, label] = phi[idx_, label]
        logits *= self.scale_factor  # scale up in order to make softmax work, first introduced in normface

        return logits

        
class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, prediction, labels):
        logp = self.ce(prediction, labels)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class SegVerID(nn.Module):
    def __init__(self, encoder = 'resnet50', decoder = 'deeplab', num_seg_classes = 19, num_id_classes = 10000, missing_labels_mask = False):
        super(SegVerID, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.missing_labels_mask = missing_labels_mask

        # set number of classes for segmentation and ID
        self.num_seg_classess = num_seg_classes
        self.num_id_classes = num_id_classes

        if self.decoder == 'unet':
            self.net = smp.Unet(self.encoder, classes = num_seg_classes)
        elif self.decoder == 'unetpp':
            self.net = smp.UnetPlusPlus(self.encoder, classes = num_seg_classes)
        elif self.decoder == 'deeplab':
            self.net = smp.DeepLabV3Plus(self.encoder, classes = num_seg_classes)

        # average pool and flatten layers to generate embeddings
        self.avg_pool2D = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)            

        # get the number of embeddings of the utilized encoder
        self.num_embeddings = self.net.encoder.out_channels[-1]            

        # init the loss classes
        if num_id_classes < 100:
            self.ArcMargin = ArcMarginProduct(scale_factor=30., margin=0.5, num_embeddings = self.num_embeddings, num_classes = self.num_id_classes)
        else:
            self.ArcMargin = ArcMarginProduct(scale_factor=64., margin=0.5, num_embeddings = self.num_embeddings, num_classes = self.num_id_classes)
        #self.ArcMargin = ArcMarginProduct()
        self.FocalLoss = FocalLoss()

        # set weight for each task's loss
        # self.seg_weight = 0.9
        # self.id_weight = 0.1

        # softmax activation for the ID task
        # self.softmax = nn.Softmax(dim = 1)

    def forward(self, input):
        # get encoder features
        features = self.net.encoder(input['image'])

        # find decoder features
        decoder_output = self.net.decoder(*features)
        
        # find masks for the batch
        seg_mask = self.net.segmentation_head(decoder_output)

        # get embeddings from final features layer of the encoder
        embeddings = self.avg_pool2D(features[-1])

        # flatten output
        embeddings = self.flatten(embeddings)
        
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return {'seg_mask' : seg_mask, 'id' : embeddings}

    def embeddings(self, input):
        # get encoder features
        features = self.net.encoder(input['image'])

        # get embeddings from final features layer of the encoder
        embeddings = self.avg_pool2D(features[-1])

        # flatten output
        embeddings = self.flatten(embeddings)

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings        
    

    def loss(self, prediction, labels):

        # mask out missing labels if necessary
        if self.missing_labels_mask:
            prediction['seg_mask'], labels['seg_mask']= sample_valid_labels(prediction['seg_mask'], labels['seg_mask'])
            prediction['id'], labels['id'] = sample_valid_labels(prediction['id'], labels['id'])

        # compute segmentation loss
        loss_seg = F.cross_entropy(prediction['seg_mask'], labels['seg_mask'])

        # calculate the ID embeddings loss
        logits = self.ArcMargin(prediction['id'], labels['id'])
        loss_ID = self.FocalLoss(logits, labels['id'])

        # find the combined weighted loss
        if loss_seg > 0.5:
            loss = loss_ID + 10*loss_seg
        if loss_seg > 0.3:
            loss = loss_ID + 5*loss_seg
        else:
            loss = loss_ID + 2.5*loss_seg

        # combined loss for fixed task loss weights
        # loss = (loss_seg * self.seg_weight) + (loss_ID * self.id_weight)

        return loss
        
    # add custom logic for the score functionality
    def accuracy(self, prediction, labels):
        # mask out missing labels if necessary
        if self.missing_labels_mask:
            prediction['id'], labels['id'] = sample_valid_labels(prediction['id'], labels['id'])
            prediction['seg_mask'], labels['seg_mask']= sample_valid_labels(prediction['seg_mask'], labels['seg_mask'])

        # convert the predictions into a score
        mask_logits = torch.argmax(prediction['seg_mask'], axis = 1)

        # calculate the ID embeddings accuracy
        logits = self.ArcMargin(prediction['id'], labels['id'])
        logits = torch.argmax(logits, axis = 1)
        acc_ID = labels['id'] == logits

        # calculate the segmentation accuracy
        acc_seg = mask_logits == labels['seg_mask']
        acc_seg = torch.sum(acc_seg, dim= [1,2]).float()/(acc_seg.shape[1]*acc_seg.shape[2])

        acc = {'seg_mask' : acc_seg, 'id' : acc_ID}

        return acc