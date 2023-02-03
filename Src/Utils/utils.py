# general utilities
import os
import xml.etree.ElementTree as ET
import multiprocessing as mp 
import csv


# def crop_ledger(data_root, xml_path, data_ledger, storage) -> None:
# 	tree = ET.parse(xml_path)
# 	root = tree.getroot()

# 	print('Determining Face Context with', mp.cpu_count(), 'CPUs')

# 	csv_columns_paths = ['im_path', 'label_path']
# 	csv_columns_landmarks = ['eye_left.x', 'eye_left.y', 'eye_right.x', 'eye_right.y', 'nose.x', 'nose.y', 'mouth_left.x', 'mouth_left.y', 'mouth_right.x', 'mouth_right.y']
# 	csv_file = os.path.join(data_root, 'ledger_crops.csv')

# 	with open(csv_file, 'w') as csvfile:
# 		writer = csv.DictWriter(csvfile, fieldnames=csv_columns_paths + csv_columns_landmarks)
# 		writer.writeheader()

# 		for index, row in data_ledger.iterrows():

# 			image = cv.imread(row['im_path'])
# 			mask = cv.imread(row['label_path'])

# 			if image.shape == mask.shape:
# 				context = {'im_path': os.path.join(storage, 'images', str(index) + '.png'), 'label_path' : os.path.join(storage, 'masks', str(index) + '.png')}
# 				for landmark in csv_columns_landmarks:
# 					context[landmark] = row[landmark]
# 				writer.writerow(context)



def ledger_csv(config) -> None:

	# generate the correct paths
	xml_path = os.path.join(config.data_root, config.dataset, config.xml_ledger)
	csv_file = os.path.join(config.data_root, config.dataset, config.data_ledger)

	# parse the xml
	tree = ET.parse(xml_path)
	root = tree.getroot()

	image = {}
	label = {}

	# convert xml to a standard format
	csv_columns_paths = ['im_path', 'mask_path']
	with open(csv_file, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=csv_columns_paths)
		writer.writeheader()

		data = {}
		for child in root:

			fixed_path = child.attrib['name'].replace("\\","/")

			# image
			if child.tag == 'srcimg':
				data['im_path'] = fixed_path
			elif child.tag == 'labelimg':
				data['mask_path'] = fixed_path
				writer.writerow(data)
				data = {}