import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
import cv2 as cv
import xml.etree.ElementTree as ET
import pandas as pd



# function to map logical values to color
def decode_segmap(mask_logits, img_width, img_height, colors):

    mask_logits = np.argmax(mask_logits, axis = 0)

    # map over the values using color codes
    mask_rgb = np.zeros((img_height, img_width, 3), dtype = np.uint8)
    for y in range(img_height):
        for x in range(img_width):
            mask_rgb[y,x] = colors[mask_logits[y,x]]
        
    return mask_rgb

class mut1ny_dataset(data.Dataset):
    def __init__(self, data_root, ledger_path, split = 'train', transform = None):
        self.colors = [(0, 0, 0),(255, 0, 0),(0, 255, 0),(0, 0, 255),(128, 128, 128),(255, 255, 0),(255, 0, 255),(0, 255, 255),(255, 255, 255),(255, 192, 192),(0, 128, 128), (0, 128, 0), (128, 0, 128), (0, 64, 64)]
        self.split = split
        self.transform = transform        

        # get the data paths
        self.img_list, self.mask_list = self.get_filenames(data_root, ledger_path)

        self.img_height = 256
        self.img_width = 256
        
    def __len__(self):
        return(len(self.img_list))
    
    def __getitem__(self, idx):
        img = cv.imread(self.img_list[idx])
        img = cv.resize(img, (self.img_width, self.img_height), interpolation = cv.INTER_LINEAR)
        mask = None
        if self.split == 'train':
            mask = cv.imread(self.mask_list[idx])
            mask = cv.resize(mask, (self.img_width, self.img_height), interpolation= cv.INTER_NEAREST)
            mask = self.encode_segmap(mask)
            assert(mask.shape == (self.img_width, self.img_height))
        
        if self.transform:
            img = self.transform(img)
            assert(img.shape == (3, self.img_height, self.img_width))
        else :
            assert(img.shape == (self.img_height, self.img_width, 3))
        
        if self.split == 'train':
            return img, mask
        else :
            return img
    
    def encode_segmap(self, mask):
        '''
        Convert RGB map to logits
        '''
        semantic_map = []

        for color in self.colors:
            equality = np.equal(mask, color)
            class_map = np.all(equality, axis = -1)
            semantic_map.append(class_map)

        semantic_map = np.stack(semantic_map, axis = -1)
        semantic_map = np.argmax(semantic_map, axis = -1)

        return semantic_map
    
    def get_filenames(self, data_root, ledger_path):
        data_ledger = pd.read_csv(ledger_path, header=0, index_col = None)
        data_ledger = data_ledger.sample(frac = 1, random_state=42)

        images = []
        labels = []

        for index, row in data_ledger.iterrows():
            images.append(os.path.join(data_root, row['im_path']))
            labels.append(os.path.join(data_root, row['mask_path']))

        return images, labels