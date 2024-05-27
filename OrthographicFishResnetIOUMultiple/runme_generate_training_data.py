from Programs.Config import Config
from Programs.Aquarium import Aquarium
from Programs.programsForGeneratingFish import x_seglen_to_3d_points, addBoxes
import os
import shutil
import multiprocessing
import numpy as np
import time
from scipy.io import loadmat
import cv2 as cv

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
    #aquarium.save_image()
    aquarium.save_image_resnet()
    #aquarium.save_annotations()

homepath = Config.dataDirectory

if not os.path.exists(homepath[:-1]):
   os.makedirs(homepath[:-1])

folders = ['images','coor_2d']
subFolders = ['train','val']
for folder in folders:
  subPath = homepath + folder
  if not os.path.exists(subPath):
      os.makedirs(subPath)


def init_pool_process():
    np.random.seed()

if __name__ == '__main__':
      # multiprocessing case
      print('Process Starting')
      startTime = time.time()
      amount = Config.amountOfData
      pool_obj = multiprocessing.Pool(initializer=init_pool_process)
      amount = 5
      pool_obj.map(genData, range(0,amount))
      pool_obj.close()
      endTime = time.time()

      print('Finish Running')
      print('Average Time: ' + str((endTime - startTime)/amount))

# for idx in range(5):
#     genData(idx)
#genData(4)

debug = False
if debug:
    # # Passing in amounts
    aquarium = Aquarium(5, fishInAllViews=1, fishInEdges=0, overlapping=0)

    aquarium.draw()
    aquarium.save_image()
    aquarium.save_annotations()

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
    aquarium.save_image()
    aquarium.save_annotations()
    



