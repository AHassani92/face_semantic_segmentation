import os
import torch

class Configs(object):
	# simple flag for debugging
	DEBUG = False
	# mode = 'Train' # ['Train', 'Test']
	mode = 'Test'

	# basic paths
	repo_path = '/s/ahassa37/code/face_seg_pl/'
	data_root = '/s/ahassa37/Data/'
	dataset = 'mut1ny_head_segmentation_pro_v2'
	data_ledger = 'ledger.csv'
	xml_ledger = 'training_fixed.xml'

	test_root = os.path.join(repo_path, 'Test', '')

	DGFA_best = 'Models/DGFA_best.pth.tar'
	models_root = os.path.join(repo_path, 'models', 'checkpoints_mut1ny', '') 
	# best_check_point = 'deeplab-resnet50-epoch=12-v1.ckpt' # 'unet-inceptionv4-epoch=11-v1.ckpt' #deeplab-resnet50-epoch=09.ckpt
	best_check_point = 'mut1ny_head_segmentation_pro_v2-deeplab-resnet50-epoch=27.ckpt'

	# always use all GPUs
	num_gpus = torch.cuda.device_count()
	learning_rate = .001
	batch_size_train = 4
	batch_size_test = 64

	# model builder configs
	# architecture = 'unet'
	# architecture = 'unetpp'
	architecture = 'deeplab'
	encoder = 'resnet50'
	# encoder = 'inceptionv4'



config = Configs()
