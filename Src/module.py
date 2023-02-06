import os
import gc
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision
#import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import random

# segmap decoding
import cv2 as cv
import multiprocessing as mp
import asyncio

# Perception library assets
from Src.Networks.FaceSeg import FaceSeg
from Src.Networks.SegID import SegID
from Src.Data.Data import dataset_generator, decode_segmap
from Src.Utils.Statistics import confusion_matrix, accuracy_rates, epoch_end_extract, generate_pairwise_ledger, pairwise_distance_ledger, pairwise_accuracy
from Src.Utils.DVP import write_eval

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


# helper function to automate key parsing
def parse_data(data, keys):

    parsed = {}

    for key in keys:
        parsed[key] = data[key]
        #[data[key] for key in keys]

    return parsed

class Face_Seg(pl.LightningModule):
    def __init__(self, config):
        super(Face_Seg, self).__init__()

        # define the base models
        self.architecture = config.architecture
        self.input_keys = config.input_keys
        self.loss_keys = config.loss_keys
        self.accuracy_keys = config.accuracy_keys
        self.colors = config.colors
        self.missing_labels = config.missing_labels

        # fetch the model
        if self.architecture == 'FaceSeg':
            self.net = FaceSeg(encoder = config.encoder, decoder = config.decoder, num_classes = len(self.colors), missing_labels_mask = self.missing_labels)

        if self.architecture == 'SegID':
            self.net = SegID(encoder = config.encoder, decoder = config.decoder, num_seg_classes = len(self.colors), num_id_classes = config.datasets['num_IDs'], missing_labels_mask = self.missing_labels)

        # set the loss and accuracy methods
        self.loss = self.net.loss
        self.accuracy = self.net.accuracy

        # tensor transformations 
        # TO DO -> fix the normalization values
        self.transform = transforms.Compose([
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

        # configure the parameters
        self.cross_val = config.cross_val
        self.batch_size_train = config.batch_size_train
        self.batch_size_test = config.batch_size_test
        self.num_cpus = config.num_cpus

        # data utilities
        self.data_root = config.data_root
        self.datasets = config.datasets
        self.img_width = config.img_width
        self.img_height = config.img_height
        self.face_crop = config.face_crop
        self.location = config.location
        self.liveliness = config.liveliness
        self.synthetic = config.synthetic
        self.test_root = config.test_root
        self.experiment_name = config.experiment_name


        # define network optimizations
        self.learning_rate = config.learning_rate
        self.threshold = 0.5

        # initialize the data to null by default
        # this allows us to simplify module init and be smart about automating data loading
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # put a function pointer wrapper on forward
    # this allows us to be agnostic to network architectures
    def forward(self, x):
        return self.net(x)

    # function to initialize the data
    def init_data(self, mode = None):

        # verify mode split
        valid = {'all', 'train', 'val', 'test'}
        if mode not in valid:
            raise ValueError("results: status must be one of %r." % valid)

        # for train mode, load only train and val data
        if mode == 'train':
            return dataset_generator(data_root = self.data_root, datasets = self.datasets['datasets_train'], splits = self.datasets['splits_train'], img_width = self.img_width,  img_height = self.img_height, seg_colors = self.colors, face_crop = self.face_crop, liveliness = self.liveliness, location = self.location, synthetic = self.synthetic, missing_labels = self.missing_labels, cross_val = self.cross_val)

        elif mode == 'val':
            return dataset_generator(data_root = self.data_root, datasets = self.datasets['datasets_val'], splits = self.datasets['splits_val'], img_width = self.img_width,  img_height = self.img_height, seg_colors = self.colors, face_crop = self.face_crop, liveliness = self.liveliness, location = self.location, synthetic = self.synthetic, missing_labels = self.missing_labels, cross_val = self.cross_val)

        elif mode == 'test':
            return dataset_generator(data_root = self.data_root, datasets = self.datasets['datasets_test'], splits = self.datasets['splits_test'], img_width = self.img_width,  img_height = self.img_height, seg_colors = self.colors, face_crop = self.face_crop, liveliness = self.liveliness, location = self.location, synthetic = self.synthetic, missing_labels = self.missing_labels, cross_val = self.cross_val)


    # impliment the model forward pass on the data and report loss
    def training_step(self, batch, batch_idx):

        # calculate the loss
        net_inputs, labels, meta_data = batch

        # network inference
        inference = self.forward(parse_data(net_inputs, self.input_keys))

        # calculate the loss
        loss = self.loss(inference, parse_data(labels, self.loss_keys))

        # calculate the accuracy metrics
        label_acc = parse_data(labels, self.accuracy_keys)
        acc = self.accuracy(inference, label_acc)

        for keys in acc.keys():
            acc[keys] = float(sum(acc[keys])/self.batch_size_train)

        self.log_dict(acc, on_step=True, on_epoch = False, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size_train)
        return {'loss' : loss}
    
    # calculate the validation accuracy
    def validation_step(self, batch, batch_idx):
        """That function operation at each validation step. confusion matrixes are being appended an validaton loss and
        accuracies are being logged
        """

        # calculate the loss
        net_inputs, labels, meta_data = batch

        # network inference
        inference = self.forward(parse_data(net_inputs, self.input_keys))

        # calculate the loss
        loss_val = self.loss(inference, parse_data(labels, self.loss_keys))

        # calculate the accuracy metrics
        label_val = parse_data(labels, self.accuracy_keys)
        val_acc = self.accuracy(inference, label_val)

        for keys in val_acc.keys():
            val_acc[keys] = float(sum(val_acc[keys])/self.batch_size_train)

        self.log_dict(val_acc, on_step=True, on_epoch = False, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size_train)
        self.log('loss_val', loss_val, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size_train)
        return {'loss_val' : loss_val}


    # load test data if not already available
    # make the test directory if does not exist
    def on_test_start(self) -> None:

        # generate the Test directory if doesn't exist
        if not os.path.exists(self.test_root):
            os.makedirs(self.test_root)

        # generate a specific experiment repo if it doesn't exist:
        self.test_root = os.path.join(self.test_root, self.experiment_name)
        if not os.path.exists(self.test_root):
            os.makedirs(self.test_root)

        # define the test report
        self.test_report = os.path.join(self.test_root, self.architecture + '_eval.csv')

    def test_step(self, batch, batch_idx):

        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss_test = F.cross_entropy(out, mask, ignore_index = 250)


        #acc_test = self.accuracy(out, mask)

        if self.test_vis:
            #mask_color = self.decode_segmap(out)
            batch_idx = batch_idx*self.batch_size_test
            #mask_logits_batch = mask_logits_batch.cpu().detach()#.numpy()
        
            # multiprocessing style
            pool = mp.Pool(mp.cpu_count())
            for num, mask in enumerate(out.cpu().detach()):
                image_name = os.path.join(self.test_dir, 'test_image_' + str(batch_idx + num).zfill(4) + '.png')
                #self.decode_helper(mask, self.img_width, self.img_height, self.colors, image_name)
                pool.apply_async(self.vis_helper, args=(mask.numpy(), self.img_width, self.img_height, self.colors, image_name))

            pool.close()
            pool.join()


        self.log('loss_test', loss_test)
        return {'loss_test' : loss_test}

    # def test_epoch_end(self, outputs) -> None:
    #     """
    #     This function accumulates all the test metrics and writes them to disk in the results folder.
    #     """
        
    #     # extract the test outputs into a simple dictionary
    #     outputs = epoch_end_extract(outputs)

    #     # calculate the pairwise distance
    #     embeddings = torch.cat(outputs['embeddings'], dim=0) 
    #     labels = torch.cat(outputs['labels'], dim=0)

    #     # compute the similarity metrics
    #     self.pairwise_ledger = generate_pairwise_ledger(labels) 
    #     similarities, targets = pairwise_distance_ledger(embeddings, labels, self.pairwise_ledger, samples = 100000)

    #     # need to break out TP, FP, TN, FN into stats
    #     stats = pairwise_accuracy(similarities.cpu().numpy(), targets.cpu().numpy())

    #     self.log('acc_test', stats['accuracy'], prog_bar=True, logger=True)
    #     self.log('th_test', stats['threshold'], prog_bar=True, logger=True)

    #     # statistics = accuracy_rates(sum(outputs['TN']), sum(outputs['FN']), sum(outputs['TP']), sum(outputs['FP']))
    #     # attack_statistics = attack_class_rates(outputs)
    #     # self.log_dict(statistics, batch_size=self.batch_size_test)
    #     #self.log_dict(attack_statistics, batch_size=self.batch_size_test)

    #     # define the ledger file and store it
        
    #     statistics = {**statistics, **attack_statistics}
    #     name = 'Cross validation ' + str(self.cross_val) if self.cross_val != None else 'Base'
    #     write_eval(self.test_report, eval = statistics, name = name)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        return [opt], [sch]
    
    # trainer function to load the train dataset
    def train_dataloader(self):

        # define the dataset
        train_dataset = self.init_data(mode = 'train')
        
        return DataLoader(train_dataset, batch_size = self.batch_size_train, shuffle = True, num_workers=self.num_cpus, drop_last=True)

    # trainer function to load the val dataset
    def val_dataloader(self):

        # define the dataset
        val_dataset = self.init_data(mode = 'val')

        return DataLoader(val_dataset, batch_size = self.batch_size_train, shuffle = False, num_workers=self.num_cpus, drop_last=True)
    
    # trainer function to load the test dataset
    def test_dataloader(self):

        # define the dataset
        test_dataset = self.init_data(mode = 'test')

        return DataLoader(test_dataset, batch_size = self.batch_size_test, shuffle = False, num_workers= self.num_cpus, drop_last=True)

    # helper function to parallelize decode and writing
    def vis_helper(self, mask, img_width, img_height, colors, image_name):
        mask_color = decode_segmap(mask, img_width, img_height, colors)
        # convert output from RGB to BGR to support opencv
        mask_color = cv.cvtColor(mask_color, cv.COLOR_RGB2BGR)         
        cv.imwrite(image_name, mask_color)




class Face_ID(pl.LightningModule):
    def __init__(self, config):
        super(Face_ID, self).__init__()

        # define the base models
        self.architecture = config.architecture
        self.input_keys = config.input_keys
        self.loss_keys = config.loss_keys
        self.accuracy_keys = config.accuracy_keys
        self.missing_labels = config.missing_labels

        # fetch the model
        if self.architecture == 'Texture':
            self.net = TextureNet(encoder = config.encoder, num_classes = config.datasets['num_IDs'], missing_labels_mask = self.missing_labels)

        elif self.architecture == 'ArcFace':
            self.net = ArcFace(encoder = config.encoder, num_classes = config.datasets['num_IDs'], missing_labels_mask = self.missing_labels)

        elif self.architecture == 'SynID':
            self.net = SynID(encoder = config.encoder, num_classes = config.datasets['num_IDs'], missing_labels_mask = self.missing_labels)




        # set the loss and accuracy methods
        self.embeddings = self.net.embeddings
        self.loss = self.net.loss
        self.accuracy = self.net.accuracy

        # tensor transformations 
        # TO DO -> fix the normalization values
        self.transform = transforms.Compose([
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

        # configure the parameters
        self.cross_val = config.cross_val
        self.batch_size_train = config.batch_size_train
        self.batch_size_test = config.batch_size_test
        self.num_cpus = config.num_cpus

        # data utilities
        self.data_root = config.data_root
        self.datasets = config.datasets
        self.img_width = config.img_width
        self.img_height = config.img_height
        self.face_crop = config.face_crop
        self.location = config.location
        self.liveliness = config.liveliness
        self.synthetic = config.synthetic
        self.test_root = config.test_root
        self.experiment_name = config.experiment_name
        self.pairwise_ledger = None
        self.verification_pass = True


        # define network optimizations
        self.learning_rate = config.learning_rate
        self.threshold = 0.5

        # initialize the data to null by default
        # this allows us to simplify module init and be smart about automating data loading
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # put a function pointer wrapper on forward
    # this allows us to be agnostic to network architectures
    def forward(self, x):
        return self.net(x)

 # function to initialize the data
    def init_data(self, mode = None):

        # verify mode split
        valid = {'all', 'train', 'val', 'test'}
        if mode not in valid:
            raise ValueError("results: status must be one of %r." % valid)

        # for train mode, load only train and val data
        if mode == 'train':
            return dataset_generator(data_root = self.data_root, datasets = self.datasets['datasets_train'], splits = self.datasets['splits_train'], img_width = self.img_width,  img_height = self.img_height, face_crop = self.face_crop, liveliness = self.liveliness, location = self.location, synthetic = self.synthetic, missing_labels = self.missing_labels, cross_val = self.cross_val)

        elif mode == 'val':
            return dataset_generator(data_root = self.data_root, datasets = self.datasets['datasets_val'], splits = self.datasets['splits_val'], img_width = self.img_width,  img_height = self.img_height, face_crop = self.face_crop, liveliness = self.liveliness, location = self.location, synthetic = self.synthetic, missing_labels = self.missing_labels, cross_val = self.cross_val)

        elif mode == 'test':
            return dataset_generator(data_root = self.data_root, datasets = self.datasets['datasets_test'], splits = self.datasets['splits_test'], img_width = self.img_width,  img_height = self.img_height, face_crop = self.face_crop, liveliness = self.liveliness, location = self.location, synthetic = self.synthetic, missing_labels = self.missing_labels, cross_val = self.cross_val)


    def verify(self):
        print("Verifying model outputs expected behavior")

        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     torchsummary.summary(self.net.to(device), (3, self.img_height, self.img_width))
        

        testloader = self.test_dataloader()
        net_inputs, labels, meta_data = next(iter(testloader))

        with torch.no_grad():

            # network inference
            inference = self.forward(parse_data(net_inputs, self.input_keys))

            # calculate the loss
            loss = self.loss(inference, parse_data(labels, self.loss_keys))

            # # calculate the accuracy metrics
            label_acc = parse_data(labels, self.accuracy_keys)
            acc = self.accuracy(inference, label_acc)

            # # get the confusion matrix then convert to rates
            # metrics = confusion_matrix(acc.cpu().numpy(), torch.argmax(label_acc, axis = 1).cpu().numpy())
            # print(accuracy_rates(metrics['TN'], metrics['FN'], metrics['TP'], metrics['FP']))

    def verify_data(self, split = 'all'):

        # generate the data
        self.init_data()
        if split == 'train': self.train_dataset.verify_data()
        elif split == 'val': self.val_dataset.verify_data()
        elif split == 'test': self.test_dataset.verify_data()
        elif split == 'all':
            self.train_dataset.verify_data()
            self.val_dataset.verify_data()
            self.test_dataset.verify_data()

        else: print('Invalid split') 

    def training_step(self, batch, batch_idx):

        # calculate the loss
        net_inputs, labels, meta_data = batch

        # network inference
        inference = self.forward(parse_data(net_inputs, self.input_keys))

        # calculate the loss
        loss = self.loss(inference, parse_data(labels, self.loss_keys))

        # calculate the accuracy metrics
        label_acc = parse_data(labels, self.accuracy_keys)
        acc = self.accuracy(inference, label_acc)

        for keys in acc.keys():
            acc[keys] = float(sum(acc[keys])/self.batch_size_train)

        #self.log('acc_step', acc, on_step=True, on_epoch = False, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size_train)
        self.log('acc_avg', acc['id'], on_step=False, on_epoch = True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size_train)
        self.log_dict(acc, on_step=True, on_epoch = False, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size_train)
        return {'loss' : loss}
    
    def validation_step(self, batch, batch_idx):
        """That function operation at each validation step. confusion matrixes are being appended an validaton loss and
        accuracies are being logged
        """

        # calculate the loss
        net_inputs, labels, meta_data = batch

        # network inference
        embeddings = self.embeddings(parse_data(net_inputs, self.input_keys))

        return {'embeddings' : embeddings, 'labels' : labels['id']}

    def validation_epoch_end(self, outputs) -> None:
        """
        This function accumulates all the test metrics and writes them to disk in the results folder.
        """
        
        # extract the test outputs into a simple dictionary
        outputs = epoch_end_extract(outputs)

        # calculate the pairwise distance
        embeddings = torch.cat(outputs['embeddings'], dim=0) 
        labels = torch.cat(outputs['labels'], dim=0)

        # compute the simalirity ledger once to save one overall time
        if self.pairwise_ledger == None:
            self.pairwise_ledger = generate_pairwise_ledger(labels) 

        #similarities, targets = pairwise_distance_ledger(embeddings, labels, self.pairwise_ledger, samples = 10000)
        similarities, targets = pairwise_distance_ledger(embeddings, labels, self.pairwise_ledger, samples = 500)

        # need to delete the ledger after the first validation test cycle, crude but necessary
        if self.verification_pass:
            self.pairwise_ledger = None
            self.verification_pass = False

        stats = pairwise_accuracy(similarities.cpu().numpy(), targets.cpu().numpy())

        self.log('acc_val', stats['accuracy'], prog_bar=True, logger=True, sync_dist=True)
        self.log('th_val', stats['threshold'], prog_bar=True, logger=True, sync_dist=True)


    # load test data if not already available
    # make the test directory if does not exist
    def on_test_start(self) -> None:

        # generate the Test directory if doesn't exist
        if not os.path.exists(self.test_root):
            os.makedirs(self.test_root)

        # generate a specific experiment repo if it doesn't exist:
        self.test_root = os.path.join(self.test_root, self.experiment_name)
        if not os.path.exists(self.test_root):
            os.makedirs(self.test_root)

        # define the test report
        self.test_report = os.path.join(self.test_root, self.architecture + '_eval.csv')

    def test_step(self, batch, batch_idx):


        # calculate the loss
        net_inputs, labels, meta_data = batch

        # network inference
        inference = self.forward(parse_data(net_inputs, self.input_keys))

        # calculate the loss
        loss_test = self.loss(inference, parse_data(labels, self.loss_keys))

        # calculate the accuracy metrics
        label_acc = parse_data(labels, self.accuracy_keys)
        acc_test = self.accuracy(inference, label_acc)
        rate = float(sum(acc_test)/self.batch_size_test)

        # get the evaluation statistics
        metrics = confusion_matrix(acc_test.cpu().numpy(), torch.argmax(label_acc, axis = 1).cpu().numpy())
        # statistics = accuracy_rates(metrics['TN'], metrics['FN'], metrics['TP'], metrics['FP'])
        attack_metrics = attack_class_eval(acc_test, meta_data)

        # log the metrics
        self.log('loss_test', loss_test, prog_bar=True, batch_size=self.batch_size_test, sync_dist=True)
        # self.log_dict(statistics, batch_size=self.batch_size_test)


        # return the accuracy metrics
        metrics = {**metrics, **attack_metrics}

        return metrics


    def test_epoch_end(self, outputs) -> None:
        """
        This function accumulates all the test metrics and writes them to disk in the results folder.
        """
        
        # extract the test outputs into a simple dictionary
        outputs = epoch_end_extract(outputs)

        # calculate the pairwise distance
        embeddings = torch.cat(outputs['embeddings'], dim=0) 
        labels = torch.cat(outputs['labels'], dim=0)

        # compute the similarity metrics
        self.pairwise_ledger = generate_pairwise_ledger(labels) 
        similarities, targets = pairwise_distance_ledger(embeddings, labels, self.pairwise_ledger, samples = 100000)

        # need to break out TP, FP, TN, FN into stats
        stats = pairwise_accuracy(similarities.cpu().numpy(), targets.cpu().numpy())

        self.log('acc_test', stats['accuracy'], prog_bar=True, logger=True)
        self.log('th_test', stats['threshold'], prog_bar=True, logger=True)

        # statistics = accuracy_rates(sum(outputs['TN']), sum(outputs['FN']), sum(outputs['TP']), sum(outputs['FP']))
        # attack_statistics = attack_class_rates(outputs)
        # self.log_dict(statistics, batch_size=self.batch_size_test)
        #self.log_dict(attack_statistics, batch_size=self.batch_size_test)

        # define the ledger file and store it
        
        statistics = {**statistics, **attack_statistics}
        name = 'Cross validation ' + str(self.cross_val) if self.cross_val != None else 'Base'
        write_eval(self.test_report, eval = statistics, name = name)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2)

        return [opt], [sch]
    
    def train_dataloader(self):

        # define the dataset
        train_dataset = self.init_data(mode = 'train')
        
        return DataLoader(train_dataset, batch_size = self.batch_size_train, shuffle = True, num_workers=self.num_cpus, drop_last=True)

    def val_dataloader(self):

        # define the dataset
        val_dataset = self.init_data(mode = 'val')

        return DataLoader(val_dataset, batch_size = self.batch_size_train, shuffle = False, num_workers=self.num_cpus, drop_last=True)
    
    def test_dataloader(self):

        # define the dataset
        test_dataset = self.init_data(mode = 'test')

        return DataLoader(test_dataset, batch_size = self.batch_size_test, shuffle = False, num_workers= self.num_cpus, drop_last=True)
