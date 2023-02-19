# basic utilities
import os
import argparse
import copy


# perception repository libraries
from Configs.config import config
from Src.trainer import main


# options
MODE = ['Default', 'Train', 'Test']
TYPE = ['Default', 'Seg', 'ID']
ARCH = ['Default', 'FaceSeg', 'SegID', 'SegVerID']

# program input parser
def parse_args():
    parser = argparse.ArgumentParser(
        description='Configuration Argument Parser For Batch Operation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-m', '--mode', type=str, default='Default',
                        help='Train or Test',
                        choices=MODE)
    parser.add_argument('-t', '--type', type=str, default='Default',
                        help='Model type - must set architecture to use this.',
                        choices=TYPE)
    parser.add_argument('-a', '--arch', type=str, default='Default',
                        help='Network architecture - must set Model Type to use this.',
                        choices=ARCH)
    args = parser.parse_args()

    return args

def config_update(configuration, args):

    output = copy.deepcopy(configuration)
    if args.mode != 'Default':
        output.mode = args.mode  

    # module configuration manager
    if args.type != 'Default' and args.arch != 'Default':
        output.model_type = 'Face_' + args.type  
        output.architecture = args.arch  
        output.get_architecture_params()
    elif args.arch == 'Default' and args.type != 'Default':
        raise ValueError("Must specify model architecture when changing type.")
    elif args.arch != 'Default' and args.type == 'Default':
        raise ValueError("Must specify model type when changing architecture.")
    return output


if __name__ == '__main__':
    
    args = parse_args()
    config = config_update(config, args)

    main(config)

    # ledger_csv(config)