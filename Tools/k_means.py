import numpy as np
from sklearn.cluster import KMeans
import cv2 as cv
import os
from glob import glob
import multiprocessing as mp
import tqdm
import pandas as pd
import warnings

# helper function to encode the segmap as logits
# necessary to properly estimate uniquness 
def encode_segmap(mask):
    '''
    Convert RGB map to logits
    '''

    # seg map from mut1ny, this should be eventually updated as a variable input
    colors = [[0, 0, 0],[0, 0, 255],[0, 255, 0],[255, 0, 0],[128, 128, 128],[0, 255, 255],[255, 0, 255],[255, 255, 0],[255, 255, 255],[192, 192, 255],[128, 128, 0], [0, 128, 0], [128, 0, 128], [64, 64, 0]]

    semantic_map = []

    for color in colors:
        equality = np.equal(mask, color)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)

    semantic_map = np.stack(semantic_map, axis = -1)
    semantic_map = np.argmax(semantic_map, axis = -1)

    return semantic_map


# core function for multiprocessing:
# acquire image seg mask path and convert to logit
def process_image_mp(im_path):
    try:
        # read the image and convert to logits
        im = cv.imread(im_path)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = encode_segmap(im)

        # k means processing
        # first transform shape
        im = im.reshape((-1,1))
        im = np.float32(im)
       
        # count number of unique face classes
        num_centers = len(np.unique(im))
        
        # fit the distribution based upon number of unique values
        kmeans = KMeans(n_clusters=num_centers, n_init = 10)
        kmeans.fit(im)

        # normalize the score
        acc = -1*kmeans.score(im)/num_centers*len(im)

    # if there are bugs, default to low score for now
    except:
        acc = 0.1
        
    return acc
    
# main program
def main(data_root):

    # warnings.filterwarnings("ignore")
    # set the local database root
    proj_root = os.getcwd()

    # set the local directory
    os.chdir(data_root)

    # get the people
    people = glob("*/")

    # setup multi-processing
    pool = mp.Pool(mp.cpu_count())

    # necessary for mp with windows
    if os.name == 'nt':
        mp.freeze_support()

    # iterate through dataset and get mask paths
    rel_paths = []
    for count, person in enumerate(people):
        
        # visualizer
        print(person, count)
        
        # set the local directory
        os.chdir(os.path.join(data_root, person))

        # append the mask paths
        images = glob('*.*')
        rel_paths += [ os.path.join(person, im) for im in images]
        #paths += glob(os.path.join(data_root, person, '*.png'))


    # convert relative paths to absolute
    paths = [os.path.join(data_root,im) for im in rel_paths]

    # process them all via multi-processing map
    num_images = len(paths)
    print('Total image masks:', num_images)
    accuracies = list(tqdm.tqdm(pool.imap(process_image_mp, paths), total=len(paths)))
    pool.close()

    # append the data ledger with these values
    print(np.mean(accuracies), np.min(accuracies), np.max(accuracies))
        
    # reset the root directory
    os.chdir(proj_root)

    headers = ['mask_path', 'mask_weight']
    df = pd.DataFrame(list(zip(rel_paths, accuracies)), columns =headers)
    df.to_csv(os.path.join(data_root, 'ledger.csv'))

if __name__ == "__main__":

    # data_root = 'F:\\Data\\LFW Labels - mut1ny_head_segmentation_pro_v2-deeplab-resnet50-epoch=27\\lfw'
    data_root = '/s/ahassa37/code_academic/face_semantic_segmentation/Test/'
    data_root = '/s/ahassa37/code_academic/face_semantic_segmentation/Test/Iterative_Annotation_v1/lfw'

    main(data_root)