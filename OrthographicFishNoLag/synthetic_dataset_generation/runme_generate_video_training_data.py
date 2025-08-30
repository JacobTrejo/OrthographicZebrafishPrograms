"""
Render synthetic images of fish to train a YOLO model
"""
import sys
from Programs.Config import Config
from Programs.Aquarium_video import Aquarium
from Programs.programsForGeneratingFishVideos import x_seglen_to_3d_points, addBoxes
import os
import shutil
import multiprocessing
import numpy as np
import time
from scipy.io import loadmat
import cv2 as cv
import json
import pdb
from tqdm import tqdm

np.random.seed(0)
# This part is to get the indices, this part can be hardcoded #########
amount_of_boxes = 9
temp_arr = np.array(range(amount_of_boxes))
original, offset = np.meshgrid(temp_arr, temp_arr)
original += offset
original = np.remainder(original, amount_of_boxes)
indices_for_permutation = list(np.concatenate([original[rowIdx, :] for rowIdx in range(amount_of_boxes)], axis=0))
# end of getting indices ################################################

def genData(idx):
    aquarium = Aquarium(idx)
    aquarium.draw()
    aquarium.save_video()
    aquarium.save_video_annotations_COCO()

homepath = Config.dataDirectory

if not os.path.exists(homepath[:-1]):
   os.makedirs(homepath[:-1])
# # Not resting it no more because it is strange, should try looking for a better function
# else:
#    # reset it
#    shutil.rmtree(homepath)
#    os.makedirs(homepath[:-1])

folders = ['images','labels']
subFolders = ['train','val']
for folder in folders:
   subPath = homepath + folder
   if not os.path.exists(subPath):
       os.makedirs(subPath)
   for subFolder in subFolders:
       subSubPath = subPath + '/' + subFolder
       if not os.path.exists(subSubPath):
           os.makedirs(subSubPath)


def init_pool_process():
    np.random.seed()


if __debug__:
    # # Passing in amounts
    for i in range(50):
        print('Running debugging script')
        aquarium = Aquarium(i)

        aquarium.draw()
        aquarium.save_video()
        aquarium.save_video_annotations_COCO()

    # Passing in fish vectors
    # the values in a fishVect are arranged as follows seglen, plane id (1 or 2) , x vector
    # fishVectList = [[ 6.80000000e+00,  1.00000000e+00,  2.88000000e+02,  1.01000000e+02,
    #                   2.09010797e+00, -2.32833530e-01, -1.01049372e+00, -6.28785639e-01,
    #                   -3.76418699e-01, -4.89061701e-01, -2.84080779e-01, -2.25273290e-01, -3.33734275e-01],
    #                 [8.80000000e+00, 2.00000000e+00, 1.78000000e+02, 1.01000000e+02,
    #                 3.09010797e+00, -2.32833530e-01, -1.01049372e+00, -6.28785639e-01,
    #                 -3.76418699e-01, -4.89061701e-01, -2.84080779e-01, -2.25273290e-01, -3.33734275e-01]]
    pi = np.pi

    # aquarium = Aquarium(0, fishVectList = fishVectList)
    aquarium.draw()
    aquarium.save_video()
    aquarium.save_video_annotations_COCO()


if __name__ == '__main__':
     # multiprocessing case
     print('Process Starting')
     startTime = time.time()
     amount = Config.amountOfData
     pool_obj = multiprocessing.Pool(initializer=init_pool_process)
     pool_obj.map(genData, range(0,amount))
     pool_obj.close()
     endTime = time.time()
     print('Finish Running')
     

     print('Compiling annotations')
     videos = []
     annotations = []

     for mode in ['train', 'val']:
        annotationsPath = os.path.join(Config.dataDirectory, 'labels', mode)
        annotationsFiles = os.listdir(annotationsPath)
        for jsonFile in tqdm(annotationsFiles):
            jsonFilePath = os.path.join(annotationsPath, jsonFile)
            with open(jsonFilePath, 'r') as f:
                dataset = json.load(f)
            videos.append(dataset['videos'])
            for annotation in dataset['annotations']:
                annotations.append(annotation)
        categories = dataset['categories']
        dataset = {
            "info": {"description": "Danionella/Zebrafish video dataset"}, 
            "videos": videos,
            "annotations": annotations,
            "categories": categories,
        }
        with open(f'{annotationsPath}/{mode}.json', 'w') as f:
            json.dump(dataset, f)
     print('Average Time: ' + str((endTime - startTime)/amount))
