import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
import cv2 as cv
import pandas as pd
import random
from torch.nn import functional as F
import math


# function to map logical values to color
def decode_segmap(mask_logits, img_width, img_height, colors):

    mask_logits = np.argmax(mask_logits, axis = 0)

    # map over the values using color codes
    mask_rgb = np.zeros((img_height, img_width, 3), dtype = np.uint8)
    for y in range(img_height):
        for x in range(img_width):
            mask_rgb[y,x] = colors[mask_logits[y,x]]
        
    return mask_rgb

class dataset_generator(data.Dataset):
    def __init__(self, data_root, datasets, splits, img_width,  img_height, seg_colors = [], face_crop = 'no', liveliness ='all', location = 'all', synthetic = 'no', missing_labels = False, ledger = 'ledger', cross_val = None, transform = None):
        
        # initialize transformations
        # by default must convert to tensor
        if transform == None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        # verify value split
        valid = {'all', 'train', 'val', 'test'}
        for split in splits:
            if split not in valid:
                raise ValueError("results: status must be one of %r." % valid)

        cross_val_print = 'cross validation - ' +str(cross_val) if cross_val != None else ''
        print('Data loader init:', data_root, datasets, splits, cross_val_print)

        # get the data ledgers
        self.split = split
        self.face_crop = face_crop
        self.liveliness = liveliness
        self.location = location
        self.synthetic = synthetic
        self.missing_labels = missing_labels
        self.colors = seg_colors

        # initiate the dataset containers
        self.im_dataset = []
        self.im_paths = []
        self.im_face_bboxes = []
        self.im_labels_id = []
        self.im_labels_liveliness = []
        self.im_labels_attack_class = []
        self.im_labels_location = []
        self.im_labels_synthetic = []
        self.seg_mask_paths = []
        self.num_IDs = 0
        self.ID_map = {}

        # parse and load the dataset containers
        for num, dataset in enumerate(datasets):
            self.parse_datasets(os.path.join(data_root, dataset), splits[num], cross_val)

        # update the length
        self.data_len = len(self.im_labels_id)

        # data loading
        self.img_width = img_width
        self.img_height = img_height

        # for CDCN network
        self.depth_map_size = [32, 32]
        self.label_weight = 0.99
        
    def __len__(self):

        return(self.data_len)
    
    def __getitem__(self, idx):

        # get the image
        im_path = self.im_paths[idx]
        img = cv.imread(im_path)

        # if we have a bbox AND flag is true, crop face
        face_bbox = self.im_face_bboxes[idx]
        if self.face_crop == 'yes' and face_bbox != None:
            img = img[face_bbox['face_top']: face_bbox['face_bottom'], face_bbox['face_left']: face_bbox['face_right']]

        # resize and convert to RGB
        img = cv.resize(img, (self.img_width, self.img_height))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # apply tensormations
        img = self.transform(img)
        assert(img.shape == (3, self.img_height, self.img_width))

        # get the basic labels
        # note that ID is offset by dataset map to avoid collisions
        label_id = self.im_labels_id[idx] + self.ID_map[self.im_dataset[idx]]
        label_liveliness = self.im_labels_liveliness[idx]
        label_synthetic = self.im_labels_synthetic[idx]

        if self.seg_mask_paths[idx] != None:
            mask = cv.imread(self.seg_mask_paths[idx])
            mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
            mask = cv.resize(mask, (self.img_width, self.img_height), interpolation= cv.INTER_NEAREST)
            mask = self.encode_segmap(mask)
            assert(mask.shape == (self.img_width, self.img_height))

        # base outputs
        net_inputs = {'image' : img}
        labels = {'id' : label_id, 'liveliness' : label_liveliness, 'synthetic' : label_synthetic, 'seg_mask' : mask}

        # extra the meta data
        attack_class = self.im_labels_attack_class[idx]
        location = self.im_labels_location[idx]

        meta_data = {'im_path' : im_path, 'attack_class' : attack_class, 'location' : location}

        return net_inputs, labels, meta_data
    
    # generate all the image paths
    def parse_datasets(self, dataset, split, cross_val):

        # get the data ledger
        ledger = 'ledger.csv' if cross_val == None else 'ledger_cv_' + str(cross_val).zfill(2) + '.csv'
        data_ledger = pd.read_csv(os.path.join(dataset, ledger), header=0, index_col = None)
        columns = data_ledger.head()

        # track the number of unique IDs including the ID map to ensure uniqueness
        self.ID_map[dataset] = self.num_IDs

        # hack to fix mut1ny, fix later
        unique = data_ledger['id'].unique()
        self.num_IDs += np.max(unique) + 1

        # get the appropraite split
        if split != 'all': 
            data_ledger = data_ledger.loc[data_ledger['split'] == self.split ]

        # filter by location        
        if self.location != 'all': 
            data_ledger = data_ledger.loc[data_ledger['location'] == self.location ]

        # filter by liveliness        
        if self.liveliness != 'all': 
            data_ledger = data_ledger.loc[data_ledger['liveliness'] == self.liveliness ]

        # filter out synthetic      
        if self.synthetic == 'no':
            if 'synthetic' in columns:
                data_ledger = data_ledger.loc[data_ledger['synthetic'] == 'real' ]

        # iterate through the dataset and generate the path and label lists
        for index, row in data_ledger.iterrows():

            # quick hack to avoid masking
            if not self.missing_labels and row['id'] == -1:
                continue         

            # hack to toss images that fail detection
            if 'face_detected' in row and row['face_detected'] == -1:
                continue

            self.im_dataset.append(dataset)
            self.im_paths.append(os.path.join(dataset, row['image_path']))
            self.im_labels_id.append(row['id'])
            self.im_labels_liveliness.append(1 if row['liveliness'] == 'live' else 0)

                
            # bounding box annotations
            #if 'face_left' in row and 'face_right' in row and 'face_top' in row and 'face_bottom' in row:
            if 'face_left' in row and not math.isnan(row['face_left']):
                bbox = {'face_left' : int(row['face_left']), 'face_right' : int(row['face_right']), 'face_top' : int(row['face_top']), 'face_bottom' : int(row['face_bottom'])}
            else:
                bbox = None

            self.im_face_bboxes.append(bbox)

            # ledger specific information
            self.seg_mask_paths.append(row['mask_path'] if 'mask_path' in columns else None)
            self.im_labels_attack_class.append(row['attack_class'] if 'attack_class' in columns else 'Unknown')
            self.im_labels_location.append(row['location'] if 'location' in columns else 'Unknown')
            self.im_labels_synthetic.append(0 if 'synthetic' in columns and row['synthetic'] == 'synethic' else 1)


    def verify_data(self):
        for k in range(self.data_len):
            data = {'image_path' : self.im_paths[k], 'id' : self.im_labels_id[k], 'liveliness' : self.im_labels_liveliness[k], 'attack_class' : self.im_labels_attack_class[k],
                'location' : self.im_labels_location[k]}
            print(data)


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


