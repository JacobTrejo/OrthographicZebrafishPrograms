import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
from PIL import Image
#import cv2
import torchvision.transforms as transforms
import pdb

class CustomImageDataset(Dataset):
    def __init__(self, img_files_address, pose_files_address, transform=None):
        self.img_files_address = img_files_address
        self.pose_files_address = pose_files_address
        self.data_size = len(img_files_address)
        self.transform = transform
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image = Image.open(self.img_files_address[idx])
        try:
            pose = torch.load(self.pose_files_address[idx])
        except:
            print(idx, ' BLAME')
        w, h = image.size
        pose[0,:] = pose[0,:] + torch.tensor(int((101 - w)/2))
        pose[1,:] = pose[1,:] + torch.tensor(int((101 - h)/2))
        image = self.transform(image)
        return image,pose
