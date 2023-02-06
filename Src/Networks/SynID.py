# torch utilities
import math
import torch
import torchvision
import torch.nn as nn
#import torch.nn.functional as F

# repository helper functions
from Src.Utils.Statistics import sample_valid_labels

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
#import pdb


"""
Implementaiton of the ArcFace loss function
See official repository: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
"""

class Flatten(Module):
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

# arcface model function
class SynID(nn.Module):
    def __init__(self, encoder = 'ResNet50', num_classes = 10000, missing_labels_mask = False):
        super(SynID, self).__init__()

        # verify it's a valid encoder 
        assert encoder in ['ResNet18', 'ResNet50', 'ResNet101', 'MobileNetV3', 'InceptionV3', 'InceptionResNet']

        # ArcFace feature generator
        # Note we only want the embeddings here, no additional fully-connected layers
        if encoder == 'MobileNetV3':
            self.feature_generator = MobileNetV3(pretrained = 'yes', model_variant = 'large')
        
        elif encoder == 'ResNet18':
            self.feature_generator = ResNet(pretrained = 'yes', model_variant = 18)

        elif encoder == 'ResNet50':
            self.feature_generator = ResNet(pretrained = 'yes', model_variant = 50)

        elif encoder == 'ResNet101':
            self.feature_generator = ResNet(pretrained = 'yes', model_variant = 101)

        elif encoder == 'InceptionV3':
            self.feature_generator = InceptionV3(pretrained = 'yes')

        elif encoder == 'InceptionResNet':
            self.feature_generator = InceptionResNet(pretrained = 'yes', model_variant = 'v2')


        # automatically fetch the number of embeddings
        self.num_embeddings = self.feature_generator.num_embeddings

        # determine batch norm layer
        self.bn1d = nn.BatchNorm1d(self.num_embeddings)

        # init the loss classes
        self.num_classes = num_classes
        if num_classes < 100:
            self.ArcMargin = ArcMarginProduct(scale_factor=30., margin=0.5, num_embeddings = self.num_embeddings, num_classes = self.num_classes)
        else:
            self.ArcMargin = ArcMarginProduct(scale_factor=64., margin=0.5, num_embeddings = self.num_embeddings, num_classes = self.num_classes)
        #self.ArcMargin = ArcMarginProduct()
        self.FocalLoss = FocalLoss()

        # determine if masking is necessary
        self.missing_labels_mask = missing_labels_mask 

        self.num_syn_classes = 2
        self.syn_classifer = nn.Sequential(
                nn.ReLU(),   
                nn.Linear(self.num_embeddings, self.num_syn_classes),
                nn.Dropout(0.3)
                )

        self.softmax = nn.Softmax(dim = 1)



    def forward(self, input):

        features = self.feature_generator(input['image'])

        # ID embeddings
        embeddings = self.bn1d(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # synthetic classification
        synthetic = self.syn_classifer(features)

        return {'id' : embeddings, 'synthetic' : synthetic}


    def embeddings(self, input):

        embeddings = self.feature_generator(input['image'])
        embeddings = self.bn1d(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def loss(self, prediction, labels):

        # mask out missing labels if necessary
        if self.missing_labels_mask:
            prediction['id'], labels['id'] = sample_valid_labels(prediction['id'], labels['id'])
            prediction['synthetic'], labels['synthetic'] = sample_valid_labels(prediction['synthetic'], labels['synthetic'])

        # calculate the ID embeddings loss
        logits = self.ArcMargin(prediction['id'], labels['id'])
        loss_ID = self.FocalLoss(logits, labels['id'])

        # calculate the synethetic loss
        score = self.softmax(prediction['synthetic'])
        loss_synthetic = F.cross_entropy(score, labels['synthetic'])

        if loss_synthetic > 0.5:
            loss = loss_ID + 10*loss_synthetic
        if loss_synthetic > 0.3:
            loss = loss_ID + 5*loss_synthetic
        else:
            loss = loss_ID + 2.5*loss_synthetic

        return loss

    # add custom logic for the score functionality
    def accuracy(self, prediction, labels):
        
        # mask out missing labels if necessary
        if self.missing_labels_mask:
            prediction['id'], labels['id'] = sample_valid_labels(prediction['id'], labels['id'])
            prediction['synthetic'], labels['synthetic'] = sample_valid_labels(prediction['synthetic'], labels['synthetic'])

        # calculate the ID embeddings accuracy
        logits = self.ArcMargin(prediction['id'], labels['id'])
        logits = torch.argmax(logits, axis = 1)

        # convert the predictions into a score
        score_synthetic = self.softmax(prediction['synthetic'])
        score_synthetic = torch.argmax(score_synthetic, axis = 1)

        acc_ID = labels['id'].cpu() == logits.cpu()
        acc_synthetic = labels['synthetic'].cpu() == score_synthetic.cpu()

        acc = {'id' : acc_ID, 'synthetic' : acc_synthetic}

        return acc
