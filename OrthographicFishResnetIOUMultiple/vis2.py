import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import torch

name = '000004'

imName = 'data/images/im_' + name + '.png'
im = cv.imread(imName)
coorName = 'data/coor_2d/ann_' + name + '.pt'
pts = torch.load(coorName)
pts = pts.numpy()

# im = cv.imread('test.png')
# pts = np.load('pt.npy')

fig = plt.figure()
ax = fig.gca()
ax.imshow(im)
plt.scatter(pts[0,:10], pts[1,:10], c = 'g', s = 50)
plt.scatter(pts[0,10:12], pts[1,10:12], c = 'r', s = 50)
plt.scatter(pts[0,12:22], pts[1,12:22], c = 'g', s = 50)
plt.scatter(pts[0,22:], pts[1,22:], c = 'r', s = 50)

plt.show()




