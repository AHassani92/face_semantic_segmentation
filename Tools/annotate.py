""" Face Annotator script

This script allows the user to label and segment facial features of images
available in either dataset or one image.

This script expects a pytorch lightning checkpoint model with full weights and model
information. It can be used with either a dataset or just one image for testing.

If dataset option is chosen, then the visualization option is not supported.
When the save option is enabled, the script would create a dataset labels folder
with the same directories structure/heirarchy as the main datasets. 

Output is saved as png with no compression to retain segmentation information 
without loss.

"""

from Config import annotate_config
import numpy as np
import argparse
from glob import glob
from Src.data import decode_segmap
from Src.module import SegModel
import os
import torch
import cv2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import math

# argument parser functions to load model and choose the annotator settings
def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False,
        help="path to the input image")
    ap.add_argument("-d", "--dataset", required=False,
        help="path to the dataset") 
    ap.add_argument("-o", "--output_path", required=False,
        help="output path for generated labels", default="labels/")
    ap.add_argument("-m", "--model", required=True,
        help="path to model for inference")
    ap.add_argument("-s", "--save_labels", required=False, default=False,
        action="store_true", help="flag to enable saving labels to output path")
    ap.add_argument("-v", "--visualize", required=False, default=False,
        action="store_true", help="flag to enable visualizing the resulting masks")
    args = vars(ap.parse_args())
    return args

# function to preprocess image when loading an individual one
def preprocess_image(image):
    # swap the color channels from BGR to RGB, resize it, and scale
    # the pixel values to [0, 1] range
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (annotate_config.IMAGE_SIZE, annotate_config.IMAGE_SIZE))
    image = image.astype("float32") / 255.0
    # subtract ImageNet mean, divide by ImageNet standard deviation,
    # set "channels first" ordering, and add a batch dimension
    image -= annotate_config.MEAN
    image /= annotate_config.STD
    image = np.transpose(image, (2, 0, 1)) # channels-frst ordering (default channel ordering expected by PyTorch)
    image = np.expand_dims(image, 0)
    # return the preprocessed image
    return image

# helper function to create an output dataset and subfolders directories to match the original structure
def create_dataset_dirs(dataset_path, out_labels_path):
    # create main labels output directory 
    try:
        os.makedirs(out_labels_path, exist_ok = True)
    except OSError as error:
        print("Directory '%s' can not be created" %out_labels_path)
        exit()            

    # get all dataset subfolder paths
    sub_folders = glob(os.path.join(dataset_path, "*"))

    # create output subfolders for each in the dataset
    for sub_folder_path in sub_folders:
        # convert to Linux dir format
        sub_folder_path = sub_folder_path.replace("\\", '/')
        # extract dataset name and subfolder name
        (dataset_name, sub_folder) = sub_folder_path.split('/')[-2:]
        # generate label subfolder output path
        out_sub_path = os.path.join(out_labels_path, dataset_name, sub_folder)

        # create a parallel subfolder in the labels output directory
        try:
            os.makedirs(out_sub_path, exist_ok = True)
        except OSError as error:
            print("Directory '%s' can not be created" %out_labels_path)
            exit()

# function to get dataset name (avoids issues due to extra slashes '/')
def get_dataset_name(dataset_path):
    # list directories and files within datasetpath
    sample_folder_path = glob(os.path.join(dataset_path, "*"))[0]

    # convert to Linux dir format
    sample_folder_path = sample_folder_path.replace("\\", '/')

    # extract dataset name
    dataset_name = sample_folder_path.split('/')[-2]

    return dataset_name


# Overridden ImageFolder class that provides file paths as well
class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    source: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path    

# main function that reads arguments provided by the user and act accordingly
def main():
    # read input arguments
    args = parse_args()

    # load model if provided. Otherwise script terminates.
    if args["model"]:
        # load the segmentation model's network and weights from disk, transfer it to the preferred device
        print("loading model: {}...".format(args["model"]))
        model = SegModel.load_from_checkpoint(args["model"]).to(annotate_config.DEVICE)

        # set model to evaluation mode
        model.eval()

        # check if image option is selected
        if args["image"]:
            image_path = args["image"]

            # load image from disk
            image = cv2.imread(args["image"])

            # preprocess image
            image = preprocess_image(image)

            # convert the preprocessed image to a torch tensor and flash it to the current device
            image = torch.from_numpy(image)
            image = image.to(annotate_config.DEVICE)            

            # run inference on image
            logits = model(image)

            # visualize mask by converting logits to RGB mask image
            mask_img = decode_segmap(logits[0].cpu().detach().numpy(), model.img_width, model.img_height, model.colors)

            # convert from RGB to BGR to support opencv
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)

            # create necessary dirs and save mask image if flag enabled
            if args["save_labels"] == True:            

                # create folder for labels output
                out_labels_path = args["output_path"]

                try:
                    os.makedirs(out_labels_path, exist_ok = True)
                except OSError as error:
                    print("Directory '%s' can not be created" %out_labels_path)
                    exit()

                # create sub-folder for image if exists
                image_dir = image_path.split('/')[-2]
                # read image name and change output image's extension to png (for no compression/quality loss)
                image_name = image_path.split('/')[-1].split('.')[0] + ".png"
                # output subfolder path
                out_subfolder_path = os.path.join(out_labels_path, image_dir)
                try:
                    os.makedirs(out_subfolder_path, exist_ok = True)
                except OSError as error:
                    print("Directory '%s' can not be created" %out_labels_path)
                    exit()
                
                # resize generated mask to original image size with no interpolation
                mask_img = cv2.resize(mask_img, ((annotate_config.ORG_IMAGE_SIZE, annotate_config.ORG_IMAGE_SIZE)), interpolation=cv2.INTER_NEAREST)
                
                # save mask to file with no/minimal compression
                cv2.imwrite(os.path.join(out_subfolder_path, image_name), mask_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # display output if flag enabled
            if args["visualize"] == True:
                cv2.imshow("mask", mask_img)
                cv2.waitKey(0)
                cv2.destroyWindow("mask")

        # check if dataset option is selected
        elif args["dataset"]:
            # get dataset path
            dataset_path = args["dataset"]

            # get dataset name from given directory
            dataset_name = get_dataset_name(dataset_path)

            # create necessary dirs and save mask image if flag enabled
            if args["save_labels"] == True:

                # get desired output labels path
                out_labels_path = args["output_path"]

                # create folder for the dataset's labels output
                create_dataset_dirs(dataset_path, out_labels_path)

                # compose necessary data preprocessing functions
                data_transforms = transforms.Compose([
                        transforms.Resize(annotate_config.IMAGE_SIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(annotate_config.MEAN, annotate_config.STD)
                    ])             
                
                # load image dataset from folder using a modified version of Pytorch's ImageFolder 
                # dataset = ImageFolder(root=dataset_path, transform=data_transforms)
                dataset = ImageFolderWithPaths(root=dataset_path, transform=data_transforms)

                # create a dataloader for the dataset
                data_loader = DataLoader(dataset, batch_size=annotate_config.BATCH_SIZE, shuffle=False)

                # get all batches from the dataset and annotate each image
                for i, (images, label, image_paths) in enumerate(data_loader):
                    print("annotating batch [", i+1, "/", math.ceil(len(dataset)/annotate_config.BATCH_SIZE), "]")                    
                    
                    # move model to preferred device
                    images = images.to(annotate_config.DEVICE)

                    # run inference on batch
                    logits = model(images)
                    
                    for i, mask in enumerate(logits.cpu().detach()):
                        # visualize mask by converting logits to RGB mask image
                        mask_img = decode_segmap(mask.numpy(), model.img_width, model.img_height, model.colors)

                        # resize generated mask to original image size with no interpolation
                        mask_img = cv2.resize(mask_img, ((annotate_config.ORG_IMAGE_SIZE, annotate_config.ORG_IMAGE_SIZE)), interpolation=cv2.INTER_NEAREST)

                        # convert from RGB to BGR to support opencv
                        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)                        

                        # get image folder name
                        sub_folder = dataset.classes[label[i]]

                        # get image name
                        image_path = image_paths[i]

                        # convert to Linux dir format
                        image_path = image_path.replace("\\", '/')

                        # extract image name and change extension to .png
                        image_name = image_path.split('/')[-1].split('.')[0] + ".png"

                        # save mask to file with no/minimal compression
                        cv2.imwrite(os.path.join(out_labels_path, dataset_name, sub_folder, image_name), mask_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])                   
    
    # if not model provided, terminate script
    else:
        exit()

if __name__ == "__main__":
    main()