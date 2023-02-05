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
from Src.module import Face_Seg

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import ssl

def main(config):
    ssl._create_default_https_context = ssl._create_unverified_context
    model = Face_Seg(config)

    # auto save if best model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath = 'models/checkpoints_mut1ny/', #checkpoints is for KITT
        filename= config.best_path,
        save_top_k = 1,
        verbose = True, 
        monitor = 'loss_val',
        mode = 'min'
    )

    # stop if loss has not improved for 5 epochs
    early_stopping = EarlyStopping(
        monitor="loss_val",
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

