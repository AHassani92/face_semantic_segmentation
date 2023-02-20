# general utilities
import os
import argparse

# repository helper functions
from Configs.config import config
from Src.Data.lfw import lfw_dataset
from Src.Data.mut1ny import mut1ny_dataset

#from Perception_Src.Utils.data_transforms import cross_validation_generator, face_box_annotator, dataset_split_resample

DATASETS_CHOICE = [0, 1, 2, 3, 4, 5, 6]
LEDGER = ['Y', 'N']
DETECT = ['Y', 'N']
CROSS_VAL = ['Y', 'N']

# program input parser
def parse_args():
    parser = argparse.ArgumentParser(
        description='Configuration Argument Parser For Batch Operation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-d', '--dataset', type=int, default=0,
                        help='Options: mut1ny, lfw',
                        choices=DATASETS_CHOICE)
    parser.add_argument('-l', '--ledger', type=str, default='N',
                        help='Generate dataset ledger.',
                        choices=LEDGER)
    parser.add_argument('-fd', '--face_detect', type=str, default='N',
                        help='Applying RetinaFace detection to dataset.',
                        choices=DETECT)
    parser.add_argument('-r', '--resample', type=str, default='N',
                        help='Resample the train, val, test splits.',
                        choices=DETECT)
    parser.add_argument('-cv', '--cross_val', type=str, default='N',
                        help='Use cross validation.',
                        choices=CROSS_VAL)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    datasets = ['mut1ny_head_segmentation_pro_v2', 'lfw']
    
    print('Processing', datasets[args.dataset])

    dataset = os.path.join(config.data_root, datasets[args.dataset])

    if args.dataset == 0:
        data = mut1ny_dataset(dataset)

    elif args.dataset == 1:
        data = lfw_dataset(dataset)

    if args.ledger == 'Y':
        data.verify_data()
        data.write_csv()

    if args.face_detect == 'Y':
        face_box_annotator(dataset)

    if args.cross_val == 'Y':
        cross_validation_generator(dataset)

    if args.resample == 'Y':
        dataset_split_resample(dataset, .7, .2, .1)

    #dataset_split_resample(dataset, train_split = 0.1, val_split = 0.9, test_split =  0.0)

