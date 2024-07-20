import cv2
import cv2 as cv
import torch
import tqdm
import os
import numpy as np
import h5py
import copy
from PIL import Image

import torchvision.transforms as transforms
from train4Test import init

from utils.group import HeatmapParser
import utils.img

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn


def post_process(det, mat_, trainval, c=None, s=None, resolution=None):
    mat = np.linalg.pinv(np.array(mat_).tolist() + [[0,0,1]])[:2]
    res = det.shape[1:3]
    print('det before parser: ', det.shape)
    cropped_preds = parser.parse(np.float32([det, det]))[0]
    
    #return cropped_preds

    #global preds
    #print('preds_shape:', cropped_preds.shape)
    preds = cropped_preds
    
    print('pred shape with 2: ', preds.shape)

    if len(cropped_preds) > 0:
        cropped_preds[:,:,:2] = utils.img.kpt_affine(cropped_preds[:,:,:2] * 4, mat) #size 1x16x3
        
    preds = np.copy(cropped_preds)
    ###for inverting predictions from input res on cropped to original image
    #if trainval != 'cropped':
    #    for j in range(preds.shape[1]):
    #        preds[0,j,:2] = utils.img.transform(preds[0,j,:2], c, s, resolution, invert=1)

    return preds

def inference(img, func, config, c, s):
    """
    forward pass at test time
    calls post_process to post process results
    """

    height, width = img.shape[0:2]
    center = (width/2, height/2)
    scale = max(height, width)/200
    res = (config['train']['input_res'], config['train']['input_res'])

    mat_ = utils.img.get_transform(center, scale, res)[:2]
    #inp = img/255
    inp = img
    print('inp shape: ',inp.shape)
    def array2dict(tmp):
        print('temp shape: ', tmp[0].shape)
        return {
            'det': tmp[0][:,:,:24],
        }
    func_out = func([inp])
    print('func_out: ', func_out[0].shape)
    tmp1 = array2dict(func([inp]))
    tmp2 = array2dict(func([inp[:,::-1]]))

    tmp = {}
#     for ii in tmp1:
#         tmp[ii] = np.concatenate((tmp1[ii], tmp2[ii]),axis=0)
    for ii in tmp1:
        tmp[ii] = np.concatenate((tmp1[ii], tmp1[ii]),axis=0)

    #     det = tmp['det'][0, -1] + tmp['det'][1, -1, :, :, ::-1][ds.flipped_parts['mpii']]
    det = tmp['det'][0, -1] + tmp['det'][1, -1,]
    #det = tmp['det'][0, 0] + tmp['det'][1, 0,] 
    if det is None:
        return [], []
    det = det/2

    det = np.minimum(det, 1)
    #global outDet
    #outDet = det
    return post_process(det, mat_, 'valid', c, s, res)

class padding:
    def __call__(self, image):
        w, h = image.size
        w_buffer = 101 - w
        w_left = int(w_buffer/2)
        w_right = w_buffer - w_left
        w_buffer = 101 - h
        w_top = int(w_buffer/2)
        w_bottom = w_buffer - w_top
        padding = (w_left, w_top, w_right, w_bottom)
        pad_transform = transforms.Pad(padding)
        padded_image = pad_transform(image)
        return padded_image

def runner(imgs):
    return func(0, config, 'inference', imgs=torch.Tensor(np.float32(imgs)))['preds']

def do(img, c, s):
    ans = inference(img, runner, config, c, s)
    return ans
    #if len(ans) > 0:
    #    ans = ans[:,:,:3]
    #
    ### ans has shape N,16,3 (num preds, joints, x/y/visible)
    #pred = []
    #for i in range(ans.shape[0]):
    #    pred.append({'keypoints': ans[i,:,:]})
    #return pred

criterion = nn.MSELoss(reduction='sum')

transform = transforms.Compose([padding(), transforms.Resize((256,256)), transforms.ToTensor(),  transforms.ConvertImageDtype(torch.float)])
transform2 = transforms.Compose([padding()])

parser = HeatmapParser()
func, config = init()
folder = '../OrthographicFishResnetIOUMultiple4/dataDel/images/'
    
files = os.listdir(folder)
files.sort()
im_to_add = [folder + fileName for fileName in files]
im_to_add = [im_to_add[29]]
#im_to_add = im_to_add[20:30]

folder = '../OrthographicFishResnetIOUMultiple4/dataDel/coor_2d/'
files = os.listdir(folder)
files.sort()
coor_to_add = [folder + fileName for fileName in files]
coor_to_add = [coor_to_add[29]]
#coor_to_add = coor_to_add[20:30]

totalLoss = 0
losses = []

for (imPath, coorPath) in zip(im_to_add, coor_to_add):
    image = Image.open(imPath)
    image = transform(image)
    image = image[0].numpy()
    image = np.stack((image, image, image), axis = 2)
    pred = do(image, None, None)
    predInCoorShape = np.zeros((2,24))
    predInCoorShape[0,:] = pred[0,:,0]
    predInCoorShape[1,:] = pred[0,:,1]
    
    shouldSave = True
    if shouldSave:
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(image)
        ax.scatter(predInCoorShape[0,:12], predInCoorShape[1,:12], c = 'g', s = 3 )
        ax.scatter(predInCoorShape[0,12:], predInCoorShape[1,12:], c = 'r', s = 3 )
        plt.savefig('pred.png')
    
    predInCoorShape *= (101/ 256)
    pred = predInCoorShape
    
    image = cv.imread(imPath)
    h, w = image.shape[:2]
    coor = torch.load(coorPath)
    coor = coor.numpy()
    
    padded_coor = np.zeros(coor.shape)
    padded_coor[0,:] = coor[0,:] + int((101 - w)/2)
    padded_coor[1,:] = coor[1,:] + int((101 - h)/2)
    
    if shouldSave:
        image = Image.open(imPath)
        image = transform2(image)
        image = np.asarray(image)
        
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(image, cmap = 'gray')
        ax.scatter(padded_coor[0,:12], padded_coor[1,:12], c = 'g', s = 3)
        ax.scatter(padded_coor[0,12:], padded_coor[1,12:], c = 'r', s = 3)
        plt.savefig('original.png')

    fishGT1 = padded_coor[:,:12]
    fishGT2 = padded_coor[:,12:]
    
    fish1 = np.copy(pred[:,:12])
    fish2 = np.copy(pred[:,12:])
    
    regLoss =  criterion(torch.from_numpy(fish1[:,:10]), torch.from_numpy(fishGT1[:,:10]) )
    flipLoss = criterion(torch.from_numpy(fish2[:,:10]), torch.from_numpy(fishGT1[:,:10]) )
    if flipLoss.item() < regLoss.item():
        fish1 = np.copy(fish2)
        fish2 = np.copy(pred[:,:12])
    
    # Lets get the eyes right
    fish1Flip = np.copy( fish1 )
    fish1Flip[:,10:] = np.flip( fish1Flip[:,10:], axis = 1)
    #fish1Flip[:,10:] = np.flip( fish1[:,10:], axis = 0)
    regLoss =  criterion(torch.from_numpy(fish1[:,10:]), torch.from_numpy( fishGT1[:,10:]) )
    flipLoss = criterion(torch.from_numpy(fish1Flip[:,10:]), torch.from_numpy(fishGT1[:,10:]) )
    #print('fish1: ',fish1[:,10:]) 
    #print('gt: ',fishGT1[:,10:])
    #print('flip: ', fish1Flip[:,10:])
    #print(flipLoss, regLoss)
    if flipLoss.item() < regLoss.item():
        fish1 = np.copy(fish1Flip)


    fish2Flip = np.copy( fish2 )
    fish2Flip[:,10:] = np.flip( fish2Flip[:,10:], axis = 1)
    #fish2Flip[:,10:] = np.flip( fish2[:,10:], axis = 0)
    regLoss =  criterion(torch.from_numpy(fish2[:,10:]), torch.from_numpy( fishGT2[:,10:]) )
    flipLoss = criterion(torch.from_numpy(fish2Flip[:,10:]), torch.from_numpy(fishGT2[:,10:]) )
    #print(flipLoss, regLoss)
    if flipLoss.item() < regLoss.item():
        fish2 = np.copy(fish2Flip)
    
    pred[:,:12] = fish1
    pred[:,12:] = fish2
    
    if shouldSave:
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(image, cmap = 'gray')
        ax.scatter(padded_coor[0,:11], padded_coor[1,:11], c = 'g', s = 3)
        ax.scatter(padded_coor[0,11],  padded_coor[1,11], c = 'g', s = 12)
        ax.scatter(padded_coor[0,12:], padded_coor[1,12:], c = 'r', s = 3)
        ax.scatter(padded_coor[0,23], padded_coor[1,23], c = 'r', s = 12) 
        ax.scatter(pred[0,:12], pred[1,:12], c = 'c', s = 3)
        ax.scatter(pred[0,11], pred[1,11], c = 'c', s = 12)
        ax.scatter(pred[0,12:], pred[1,12:], c = 'm', s = 3)
        ax.scatter(pred[0,23], pred[1,23], c = 'm', s = 12) 
        plt.savefig('final.png')


    loss = criterion(torch.from_numpy(pred), torch.from_numpy(padded_coor) ) 
    #print(loss.item())
    losses.append(loss.item())
    totalLoss += loss.item()

print(losses)
print(totalLoss / len(im_to_add))











