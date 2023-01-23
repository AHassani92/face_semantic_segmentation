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

from Src.data import mut1ny_dataset, decode_segmap

from torchmetrics import Accuracy
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import segmentation_models_pytorch as smp
import ssl

class SegModel(pl.LightningModule):
    def __init__(self, config):
        super(SegModel, self).__init__()
        self.batch_size = config.batch_size_train
        self.batch_size_test = config.batch_size_test
        self.train_split = .7
        self.val_split = .2
        self.test_split = .1
        self.learning_rate = 1e-3
        self.architecture = config.architecture
        self.encoder = config.encoder
        self.test_root = config.test_root
        self.test_vis = True 
        self.accuracy = Accuracy()

        # BGR color codes
        # self.colors = [[0, 0, 0],[255, 0, 0],[0, 255, 0],[0, 0, 255],[128, 128, 128],[255, 255, 0],[255, 0, 255],[0, 255, 255],[255, 255, 255],[255, 192, 192],[0, 128, 128], [0, 128, 0], [128, 0, 128], [0, 64, 64]]

        # RGB color codes
        self.colors = [[0, 0, 0],[0, 0, 255],[0, 255, 0],[255, 0, 0],[128, 128, 128],[0, 255, 255],[255, 0, 255],[255, 255, 0],[255, 255, 255],[192, 192, 255],[128, 128, 0], [0, 128, 0], [128, 0, 128], [64, 64, 0]]

        self.img_height = 256
        self.img_width = 256

        if self.architecture == 'unet':
            #self.net = UNet(num_classes = 19, bilinear = False)
            self.net = smp.Unet(self.encoder, classes = 19)
        elif self.architecture == 'unetpp':
            self.net = smp.UnetPlusPlus(self.encoder, classes = 19)
        elif self.architecture == 'deeplab':
            #self.net = smp.DeepLabV3(encoder, classes = 19)
            self.net = smp.DeepLabV3Plus(self.encoder, classes = 19)
        elif self.architecture == 'enet':
            self.net = ENet(num_classes = 19)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.35675976, 0.37380189, 0.3764753], std = [0.32064945, 0.32098866, 0.32325324])
        ])

        # get the data ledger
        self.dataset = config.dataset
        dataset_path = os.path.join(config.data_root, config.dataset)
        self.data_ledger = os.path.join(config.data_root, config.dataset, config.data_ledger)

        # get all the images
        full_train = mut1ny_dataset(data_root = dataset_path, ledger_path = self.data_ledger, split = 'train', transform = self.transform)

        # determine the training and validation splits
        train_split = range(0, int(self.train_split*len(full_train)))
        val_split = range(int(self.train_split*len(full_train)), int((self.train_split + self.val_split)*len(full_train)))
        test_split = range(int((self.train_split + self.val_split)*len(full_train)), len(full_train))

        self.trainset = Subset(full_train, train_split)
        self.valset = Subset(full_train, val_split)
        self.testset = Subset(full_train, test_split)
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_nb) :
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss = F.cross_entropy(out, mask)
        #acc = self.accuracy(out, mask)

        self.log('loss', loss, on_step=True, prog_bar=True, logger=True)
        #self.log('acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss' : loss}
    
    def validation_step(self, batch, batch_idx):
        """That function operation at each validation step. confusion matrixes are being appended an validaton loss and
        accuracies are being logged
        """

        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss_val = F.cross_entropy(out, mask, ignore_index = 250)

        self.log('loss_val', loss_val, on_epoch=True, prog_bar=True, logger=True)
        #self.log('val_acc', val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss_val' : loss_val}


    # make the test storage directory
    def on_test_start(self) -> None:
        self.test_dir_dataset = os.path.join(self.test_root, self.dataset)
        if not os.path.exists(self.test_dir_dataset):
            os.makedirs(self.test_dir_dataset)

        self.test_dir = os.path.join(self.test_dir_dataset, self.architecture + '-' + self.encoder + '/')
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

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

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        return [opt], [sch]
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True, num_workers=32, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size = self.batch_size, shuffle = False, num_workers=32, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size = self.batch_size_test, shuffle = False, num_workers= 32, drop_last=True)

    # # function to map logical values to color
    # def decode_segmap(self, mask_logits, img_width, img_height, colors):

    #     # # convert operation to CPU for loops
    #     # mask_logits_batch = mask_logits_batch.cpu().detach()#.numpy()

    #     # mask_rgb_list = []
    #     # for mask_logits in mask_logits_batch:
    #         # get the inferred logical value from the 14 channels
    #     mask_logits = np.argmax(mask_logits, axis = 0)

    #     # map over the values using color codes
    #     mask_rgb = np.zeros((img_height, img_width, 3), dtype = np.uint8)
    #     for y in range(img_height):
    #         for x in range(img_width):
    #             mask_rgb[y,x] = colors[mask_logits[y,x]]
            
    #     return mask_rgb

    # helper function to parallelize decode and writing
    def vis_helper(self, mask, img_width, img_height, colors, image_name):
        mask_color = decode_segmap(mask, img_width, img_height, colors)
        # convert output from RGB to BGR to support opencv
        mask_color = cv.cvtColor(mask_color, cv.COLOR_RGB2BGR)         
        cv.imwrite(image_name, mask_color)

def main(config):
    ssl._create_default_https_context = ssl._create_unverified_context
    model = SegModel(config)

    # auto save if best model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath = 'models/checkpoints_mut1ny/', #checkpoints is for KITT
        filename= config.dataset + '-' + config.architecture + '-' + config.encoder + '-{epoch:02d}',
        save_top_k = 1,
        verbose = True, 
        monitor = 'loss_val',
        mode = 'min'
    )

    # stop if loss has not improved for 5 epochs
    early_stopping = EarlyStopping(
        monitor="loss",
        min_delta=0.001,
        patience=10)


    # for training
    if config.mode == 'Train':
        # debug on single GPU
        if config.DEBUG:
            trainer = pl.Trainer(accelerator='gpu', devices=1, auto_select_gpus=True, max_epochs = 150, callbacks = [checkpoint_callback, early_stopping], log_every_n_steps = 1)
        else:
            trainer = pl.Trainer(accelerator='gpu', devices=config.num_gpus, auto_select_gpus=True, strategy=DDPStrategy(find_unused_parameters=False), num_nodes =1, max_epochs = 150, callbacks = [checkpoint_callback, early_stopping], log_every_n_steps = 1)
        trainer.fit(model)
    # trainer.test(ckpt_path='best')

    # for testing
    if config.mode == 'Test':
        trainer = pl.Trainer(accelerator='gpu', devices=1, auto_select_gpus=True, log_every_n_steps = 1)
        ckpt_path = os.path.join(config.models_root, config.best_check_point)
        trainer.test(model, ckpt_path=ckpt_path)


