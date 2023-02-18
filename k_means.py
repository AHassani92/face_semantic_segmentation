import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import cv2 as cv
import os
from glob import glob
import multiprocessing as mp
import tqdm

def encode_segmap(mask):
    '''
    Convert RGB map to logits
    '''
    semantic_map = []

    for color in colors:
        equality = np.equal(mask, color)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)

    semantic_map = np.stack(semantic_map, axis = -1)
    semantic_map = np.argmax(semantic_map, axis = -1)

    return semantic_map

def process_image(im_path):
    try:
        # read the image and convert to logits
        im = cv.imread(im_path)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = encode_segmap(im)
        im = im.reshape((-1,1))
        im = np.float32(im)
       
        # count number of unique face classes
        num_centers = len(np.unique(im))
        #print(np.unique(im))
        
        kmeans = KMeans(n_clusters=num_centers)
        kmeans.fit(im)
        acc = 1+kmeans.score(im)/num_centers*len(im)
        #print(acc, flush=True)
        #acc = -1*kmeans.score(im)
    except:
        print(im)
        acc = 1
        
    return acc

def log_results(result,):
    accuracies.append(result)

def process_image_mp(im_path):
    try:
        # read the image and convert to logits
        im = cv.imread(im_path)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = encode_segmap(im)
        im = im.reshape((-1,1))
        im = np.float32(im)
       
        # count number of unique face classes
        num_centers = len(np.unique(im))
        
        kmeans = KMeans(n_clusters=num_centers)
        kmeans.fit(im)
        acc = 1+kmeans.score(im)/num_centers*len(im)

    except:
        acc = 0.1
        
    return acc
    

accuracies = []
colors = [[0, 0, 0],[0, 0, 255],[0, 255, 0],[255, 0, 0],[128, 128, 128],[0, 255, 255],[255, 0, 255],[255, 255, 0],[255, 255, 255],[192, 192, 255],[128, 128, 0], [0, 128, 0], [128, 0, 128], [64, 64, 0]]


def main(data_root):
    # set the local database root
    proj_root = os.getcwd()

    # set the local directory
    os.chdir(data_root)

    people = glob("*/")
    NUM_FACE_CLASSES = len(people)

    # setup multiprocessing
    pool = mp.Pool(mp.cpu_count())
    mp.freeze_support()


    paths = []
    for count, person in enumerate(people):
        
        print(person, count)
        
        #images = glob(os.path.join(data_root, person, '*.png'))
        paths += glob(os.path.join(data_root, person, '*.png'))

    num_images = len(paths)

    # accuracies = pool.map(process_image_mp, paths)
    print('Total image masks:', num_images)
    accuracies = list(tqdm.tqdm(pool.imap(process_image_mp, paths), total=len(paths)))
    pool.close()

    print(accuracies)
        # if (count % 10) == 0:
        #     print(count, np.mean(accuracies), np.min(accuracies), np.max(accuracies))
        
    # reset the root directory
    os.chdir(proj_root)

if __name__ == "__main__":
    
    data_root = 'F:\\Data\\LFW Labels - mut1ny_head_segmentation_pro_v2-deeplab-resnet50-epoch=27\\lfw'
    main(data_root)