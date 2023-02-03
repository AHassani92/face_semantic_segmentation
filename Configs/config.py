# basic utilities
import os
import yaml
import os
import torch

class Configs(object):
	def __init__(self):
		super(Configs, self).__init__()
		# simple flag for debugging
		self.DEBUG = False
		#assert self.DEBUG in ['True', 'False']

		# determine training vs testing
		self.mode = 'Train' # 
		#mode = 'Test'
		assert self.mode in ['Train', 'Test']

		# basic paths
		self.repo_path = os.getcwd()
		self.configs_root = os.path.join(self.repo_path, 'Configs', '') 
		self.data_root = os.path.join(self.repo_path, 'Data', '') 
		self.models_root = os.path.join(self.repo_path, 'Models', '') 
		self.test_root = os.path.join(self.repo_path, 'Test', '') 


		# experiment configuration
		self.experiment_name = 'SegID_experimentation'
		self.model_type = 'Face_Seg' # ['Face_Detect', 'Face_ID', 'Face_PAD', 'Face_Seg']
		assert self.model_type in ['Face_Detect', 'Face_ID', 'Face_PAD', 'Face_Seg']

		# use datasets yaml to setup the dataset configuration
		self.get_datasets_parms()

		# specify additional constraints
		self.face_crop = 'no' # ['yes', 'no']
		self.location = 'all' # ['all', 'interior', 'exterior']
		self.liveliness = 'live' # ['all', 'live', 'spoof']
		self.synthetic = 'yes' # ['yes', 'no']
		self.missing_labels = False # use when have MTL problems and partially missing annotations, ['yes', 'no']


		# training parameters
		self.num_gpus = torch.cuda.device_count()
		self.num_cpus = 16
		self.learning_rate = .001
		self.batch_size_train = 16 # size of training batches, small for accuracy
		self.batch_size_test = 64 # size of the test batches, large for speed
		self.check_val_every_n_epoch = 1 # pytorch lightning parameter to perform validation after each check_val_every_n_epoch training epochs
		self.auto_scale_batch_size = False # pytorch lightning parameter to auto scale the batch_sie
		self.log_every_n_steps = 1 # pytorch lightning parameter to log every log_every_n_steps training steps
		self.cross_val = None # whether to randomize the datasets for cross validation, set to None or a number of validations
		self.max_epochs = 150 # maximum number of training epochs
		self.patience_epochs = 10 # how many training iterations before early stopping

		# seg parameters: color map
		self.colors = [[0, 0, 0],[0, 0, 255],[0, 255, 0],[255, 0, 0],[128, 128, 128],[0, 255, 255],[255, 0, 255],[255, 255, 0],[255, 255, 255],[192, 192, 255],[128, 128, 0], [0, 128, 0], [128, 0, 128], [64, 64, 0]]


		# model builder configs
		# architecture options: ['Texture', 'DGFA', 'CDCN', 'LGSC', 'SAIPS']
		self.architecture = 'FaceSeg' 

		# verify the architecture supports the model type
		if self.model_type == 'Face_Seg':
			assert self.architecture in ['FaceSeg']

		self.get_architecture_params()

	# helper function to read in the dataset yaml file
	# fetch total number of faces
	def get_datasets_parms(self):


		# automate this next
		# fetch the datasets yaml 
		with open(os.path.join(self.configs_root, 'Datasets.yaml'), 'r') as f:
			dataset_args = yaml.safe_load(f)

		# one config to store the dataset information
		self.datasets = {}
		self.datasets['num_IDs'] = 0

		# dataset ledgers
		self.datasets['datasets_train'] = [dataset for dataset in dataset_args['train']]
		self.datasets['datasets_val'] = [dataset for dataset in dataset_args['val']]
		self.datasets['datasets_test'] = [dataset for dataset in dataset_args['test']]

		# corresponding splits
		self.datasets['splits_train'] = [dataset_args['train'][dataset]['split'] for dataset in dataset_args['train']]
		self.datasets['splits_val'] = [dataset_args['val'][dataset]['split'] for dataset in dataset_args['val']]
		self.datasets['splits_test'] = [dataset_args['test'][dataset]['split'] for dataset in dataset_args['test']]

		# fetch the number of IDs per dataset
		#ledgers = list(set(self.datasets['datasets_train'] + self.datasets['datasets_val'] + self.datasets['datasets_test']))
		ledgers = list(set(self.datasets['datasets_train']))
		for dataset in ledgers:
			self.datasets['num_IDs'] += dataset_args['num_IDs'][dataset]

		self.num_seg_classes = 19

		# visualize datasets in debug mode
		if self.DEBUG:
			print(self.datasets)

	# helper function to read in the architecture yaml file
	# fetch model I/O and encoder
	def get_architecture_params(self):

		# fetch the model parameters from the appropriate parameters 
		with open(os.path.join(self.configs_root, 'Networks', self.architecture + '.yaml'), 'r') as f:
			configs_args = yaml.safe_load(f)

		self.input_keys = configs_args[self.model_type]['input_keys']
		self.loss_keys = configs_args[self.model_type]['loss_keys']
		self.accuracy_keys = configs_args[self.model_type]['accuracy_keys']
		self.encoder = configs_args[self.model_type]['encoder']
		self.pretrain_path = configs_args[self.model_type]['pretrain']
		self.best_path = self.model_type + '_' + self.architecture + '_' + self.encoder  +'_best'

		if self.model_type == 'Face_Seg':
			self.decoder = configs_args[self.model_type]['decoder']
			assert self.decoder in ['none', 'unet', 'unetpp', 'deeplab']
		else:
			self.decoder = None

		# set the sizes if not defined
		if configs_args[self.model_type]['img_size'] != 'None':
			self.img_width = self.img_height = configs_args[self.model_type]['img_size']

		else:
			if 'MobileNet' in self.encoder:
				self.img_width = self.img_height = 160
			elif 'Inception' in self.encoder:
				self.img_width = self.img_height = 299
			elif 'ResNet' in self.encoder:
				self.img_width = self.img_height = 224
			else:
				self.img_width = self.img_height = 256

		# visualize architecture in debug mode
		if self.DEBUG:
			print(self.architecture, self.encoder, self.decoder)

	# dataset = 'mut1ny_head_segmentation_pro_v2'
	# data_ledger = 'ledger.csv'
	# xml_ledger = 'training_fixed.xml'


	# models_root = os.path.join(repo_path, 'models', 'checkpoints_mut1ny', '') 
	# # best_check_point = 'deeplab-resnet50-epoch=12-v1.ckpt' # 'unet-inceptionv4-epoch=11-v1.ckpt' #deeplab-resnet50-epoch=09.ckpt
	# best_check_point = 'mut1ny_head_segmentation_pro_v2-deeplab-resnet50-epoch=27.ckpt'

	# # always use all GPUs
	# num_gpus = torch.cuda.device_count()
	# learning_rate = .001
	# batch_size_train = 4
	# batch_size_test = 64

	# # model builder configs
	# # architecture = 'unet'
	# # architecture = 'unetpp'
	# architecture = 'deeplab'
	# encoder = 'resnet50'
	# # encoder = 'inceptionv4'


config = Configs()
