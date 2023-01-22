""" Config file for the Face Annotator script

Make sure to set the ORG_IMAGE_SIZE to match the dataset original image size.

Only supports 1 size. Otherwise, changes to the main script are needed:
- Load image size and return as part of the batch. Update ImageFolder class.
"""

# load necessary packages
import torch

# specify image dimension
IMAGE_SIZE = 256

# specify the dataset's original image size
ORG_IMAGE_SIZE = 250

# set batch size
BATCH_SIZE = 16

# specify ImageNet mean and standard deviation (RGB)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# determine the device we will be using for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"