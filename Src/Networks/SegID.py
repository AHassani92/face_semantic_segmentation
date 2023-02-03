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
    def __init__(self, encoder = 'ResNet50', decoder = 'deeplab', num_classes = 19, missing_labels_mask = False):
        super(SegID, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.missing_labels_mask = missing_labels_mask

        if self.decoder == 'unet':
            self.net = smp.Unet(self.encoder, classes = num_classes)
        elif self.decoder == 'unetpp':
            self.net = smp.UnetPlusPlus(self.encoder, classes = num_classes)
        elif self.decoder == 'deeplab':
            self.net = smp.DeepLabV3Plus(self.encoder, classes = num_classes)
        
    def forward(self, input):
        seg_mask = self.net(input['image'])

        return {'seg_mask' : seg_mask}
    

    def loss(self, prediction, labels):

        # mask out missing labels if necessary
        if self.missing_labels_mask:
            prediction['seg_mask'], labels['seg_mask']= sample_valid_labels(prediction['seg_mask'], labels['seg_mask'])

        # compute loss
        # mask_logits = nn.Softmax(prediction['seg_mask'], dim = 1)
        # loss = F.cross_entropy(mask_logits, labels['seg_mask'])
        loss = F.cross_entropy(prediction['seg_mask'], labels['seg_mask'])


        # loss = F.cross_entropy(score, labels['seg_mask'])

        return loss
        
    # add custom logic for the score functionality
    def accuracy(self, prediction, labels):

        # convert the predictions into a score
        mask_logits = torch.argmax(prediction['seg_mask'], axis = 1)

        # calculate the accuracy
        acc = mask_logits == labels['seg_mask']
        acc = torch.sum(acc, dim= [1,2]).float()/(acc.shape[1]*acc.shape[2])

        return {'acc_mask' : acc}