import os
import xml.etree.ElementTree as ET
from glob import glob
import csv


class mut1ny_dataset():
    def __init__(self, data_root, ledger = 'ledger.csv', xml = 'training_fixed.xml'):
        super(mut1ny_dataset, self).__init__()

        self.colors = [(0, 0, 0),(255, 0, 0),(0, 255, 0),(0, 0, 255),(128, 128, 128),(255, 255, 0),(255, 0, 255),(0, 255, 255),(255, 255, 255),(255, 192, 192),(0, 128, 128), (0, 128, 0), (128, 0, 128), (0, 64, 64)]

        self.data_root = data_root
        self.ledger = ledger
        self.xml = xml

        self.participants = []
        self.im_paths = []
        self.im_labels_id = []
        self.im_labels_liveliness = []
        self.im_labels_attack_class = [] 
        self.im_labels_location =[]
        self.im_splits = []
        self.mask_paths = []
        self.im_synthetic = []

        self.get_filenames()
        
    def __len__(self):
        return(len(self.img_list))

    def get_unique_IDs(self, participants):

        unique = []
        for ID in participants:
            for ID_unique in participants:
                if ID_unique in ID:
                    unique.append(ID_unique)
                    break

        unique = list(set(unique))

        #print(unique, len(unique), len(participants))
        return unique

    def is_synthetic(self, labels_path):

        if 'real' in labels_path:
            return 'real'
        else:
            return 'synthetic'


    def get_ID(self, labels_path):

        for num, ID in enumerate(self.participants):
            if ID in labels_path:
                break

        if 'multi' in ID or 'real' in ID:
            num = -1
            #print(ID, labels_path, num)

        return num
    
    def get_filenames(self):

        # generate the correct paths
        xml_path = os.path.join(self.data_root, self.xml)

        # mut1ny is odd format, need to use labels to determine IDs
        labels_path = os.path.join(self.data_root, 'labels', '')

        # get the local database root
        proj_root = os.getcwd()
     
        # set the labels directory
        os.chdir(labels_path)

        # iterate through the directories
        self.participants = glob("*/")
        self.participants = sorted(self.participants)
        self.participants = [os.path.normpath(dir) for dir in self.participants]
        self.participants = self.get_unique_IDs(self.participants)
        self.num_participants = len(self.participants)

        # parse the xml
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # convert xml to a standard format
        data = {}
        for child in root:

            fixed_path = child.attrib['name'].replace("\\","/")

            # image
            if child.tag == 'srcimg':
                data['im_path'] = fixed_path
            elif child.tag == 'labelimg':
                data['mask_path'] = fixed_path
                ID = self.get_ID(fixed_path)
                synthetic = self.is_synthetic(fixed_path)
                #writer.writerow(data)

                self.im_paths.append(os.path.join(self.data_root, data['im_path']))
                self.mask_paths.append(os.path.join(self.data_root, data['mask_path']))


                if ID < int(.7*self.num_participants): split = 'train'
                elif ID < int(.9*self.num_participants): split = 'val'
                else: split = 'test' 

                self.im_labels_id.append(ID)
                self.im_labels_liveliness.append('live')
                self.im_labels_attack_class.append('None')
                self.im_labels_location.append('Unknown')
                self.im_splits.append(split)
                self.im_synthetic.append(synthetic)
                    
        # reset the root directory
        os.chdir(proj_root)

        self.data_len = len(self.im_paths)


    # function to verify the data is good
    def verify_data(self):
        for k in range(self.data_len):
            data = {'image_path' : self.im_paths[k],'mask_path' : self.mask_paths[k], 'id' : self.im_labels_id[k], 'liveliness' : self.im_labels_liveliness[k], 'attack_class' : self.im_labels_attack_class[k],
                'location' : self.im_labels_location[k], 'split' : self.im_splits[k], 'synthetic' : self.im_synthetic[k]}
            print(data)


    # function to take the lab 3112 data and write to csv
    def write_csv(self):

        # define the ledger file
        csv_file = os.path.join(self.data_root, self.ledger)
        csv_columns = ['image_path', 'mask_path', 'id', 'liveliness', 'attack_class', 'location', 'split', 'synthetic']

        # open a csv writer and generate the ledger
        with open(csv_file, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = csv_columns)
            writer.writeheader()

            # write each value to disk
            for k in range(self.data_len):
                data = {'image_path' : self.im_paths[k],'mask_path' : self.mask_paths[k], 'id' : self.im_labels_id[k], 'liveliness' : self.im_labels_liveliness[k], 'attack_class' : self.im_labels_attack_class[k],
                    'location' : self.im_labels_location[k], 'split' : self.im_splits[k], 'synthetic' : self.im_synthetic[k]}
                writer.writerow(data)








