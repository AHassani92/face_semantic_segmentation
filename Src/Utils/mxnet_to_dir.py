import numbers
import os

from PIL import Image
import multiprocessing as mp
import cv2 as cv
import mxnet as mx
import numpy as np

def data_write_mp(write_root, label, index, mx_img):
    
    # determine the proper directory
    directory = 'Participant_'+str(int(label)).zfill(5)
    
    # generate a specific experiment repo if it doesn't exist:
    directory = os.path.join(write_root, directory)
    if not os.path.exists(directory):
        os.makedirs(directory)   
        
    image = mx.image.imdecode(mx_img).asnumpy()
    image = Image.fromarray(image) 
    
    im_name = 'im_' + str(index).zfill(3) + '.png'
    im_path = os.path.join(directory, im_name)
    
    image.save(im_path)

# function to parse mxnet records into a  traditional data repo
def mxnext_to_dir(data_root, write_root = None):
    
    if write_root == None:
        write_root = data_root
        
    # generate the write directory to be safe
    if not os.path.exists(write_root):
        os.makedirs(write_root)
        
    # Mxnet RecordIO
    path_imgrec = os.path.join(data_root, 'train.rec')
    path_imgidx = os.path.join(data_root, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    s = imgrec.read_idx(0)

    header, _ = mx.recordio.unpack(s)
    if header.flag > 0:
        header0 = (int(header.label[0]), int(header.label[1]))
        imgidx = np.array(range(1, int(header.label[0])))
    else:
        imgidx = np.array(list(self.imgrec.keys))

    prev_label = 0
    idx_offset = 0
    
    pool = mp.Pool(mp.cpu_count())
    
    print('Processing MX Net Ledger with {} CPUS'.format(mp.cpu_count()))
    for index in range(len(imgidx)):
        idx = imgidx[index]
        s = imgrec.read_idx(idx)
        header, mx_img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        
        # smarter label indexing per person
        if prev_label != label:
            prev_label = label
            idx_offset = idx
    
        # call the multiprocessing write function
        pool.apply_async(data_write_mp,  args=(write_root, label, idx - idx_offset, mx_img))

        if index == 0 or index % 10000 == 0:
            print('Adding index {} to MP pool with label {}'.format(index, label))
                
    print('Executing pool')
    pool.close()
    pool.join()



data_root = '/s/ahassa37/Data/faces_emore'
write_root = '/s/ahassa37/Data/MS1M'


mxnext_to_dir(data_root, write_root)
