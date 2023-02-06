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

class SegID(nn.Module):
    def __init__(self, encoder = 'ResNet50', decoder = 'deeplab', num_seg_classes = 19, num_id_classes = 86, missing_labels_mask = False):
        super(SegID, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.missing_labels_mask = missing_labels_mask

        # set number of classes for segmentation and ID
        self.num_seg_classess = num_seg_classes
        self.num_id_classes = num_id_classes

        # create a dict auxilliary task parameters
        self.id_params = {}
        # add number of classes (face id classes)
        self.id_params["classes"] = num_id_classes
        # set pooling to average
        self.id_params["pooling"] = "avg"
        # set dropout
        self.id_params["dropout"] = 0.2
        # set activation to None to return logits (activation is set to softmax in loss() function)
        self.id_params["activation"] = None

        if self.decoder == 'unet':
            self.net = smp.Unet(self.encoder, classes = num_seg_classes, aux_params=self.id_params)
        elif self.decoder == 'unetpp':
            self.net = smp.UnetPlusPlus(self.encoder, classes = num_seg_classes, aux_params=self.id_params)
        elif self.decoder == 'deeplab':
            self.net = smp.DeepLabV3Plus(self.encoder, classes = num_seg_classes, aux_params=self.id_params)

        # set weight for each task's loss
        self.seg_weight = 0.9
        self.id_weight = 0.1

        # softmax activation for the ID task
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input):
        (seg_mask, id) = self.net(input['image'])

        return {'seg_mask' : seg_mask, 'id' : id}
    

    def loss(self, prediction, labels):

        # mask out missing labels if necessary
        if self.missing_labels_mask:
            prediction['seg_mask'], labels['seg_mask']= sample_valid_labels(prediction['seg_mask'], labels['seg_mask'])

        # compute loss
        loss_seg = F.cross_entropy(prediction['seg_mask'], labels['seg_mask'])

        # calculate the ID loss
        score = self.softmax(prediction['id'])
        loss_id = F.cross_entropy(score, labels['id'])

        # find the combined weighted loss
        loss = (loss_seg * self.seg_weight) + (loss_id * self.id_weight)

        return loss
        
    # add custom logic for the score functionality
    def accuracy(self, prediction, labels):
        # mask out missing labels if necessary
        if self.missing_labels_mask:
            prediction['id'], labels['id'] = sample_valid_labels(prediction['id'], labels['id'])

        # convert the predictions into a score
        mask_logits = torch.argmax(prediction['seg_mask'], axis = 1)

        # calculate the segmentation accuracy
        acc_seg = mask_logits == labels['seg_mask']
        acc_seg = torch.sum(acc_seg, dim= [1,2]).float()/(acc_seg.shape[1]*acc_seg.shape[2])

        # calculate the face ID accuracy
        score = self.softmax(prediction['id'])
        score = torch.argmax(score, axis = 1)
        # acc_ID = torch.sum(labels['id'] == score).float()/len(labels['id'])
        acc_ID = (labels['id'] == score) # Note: implemented this way to be compatible with Face_Seg's training and validation step's accuracy estimation

        acc = {'seg_mask' : acc_seg, 'id' : acc_ID}

        return acc