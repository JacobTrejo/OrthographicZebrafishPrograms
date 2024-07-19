import cv2
import sys
import os
import torch
import numpy as np
import torch.utils.data
import utils.img
from PIL import Image
import torchvision.transforms as transforms

class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma
        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[0] > 0: 
                    x, y = int(pt[0]), int(pt[1])
                    if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                        continue
                    ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                    br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                    c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms

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

class ZebrafishDataset(torch.utils.data.Dataset):
    def __init__(self, config, img_files_address, pose_files_address, transform=None):
        self.config = config
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.generateHeatmap = GenerateHeatmap(self.output_res, config['inference']['num_parts'])
        
        self.img_files_address = img_files_address
        self.pose_files_address = pose_files_address
        self.data_size = len(img_files_address)
        self.transform = transform

    def __len__(self):
        return self.data_size
    
    #def __init__(self, config, ds, index):
    #    self.input_res = config['train']['input_res']
    #    self.output_res = config['train']['output_res']
    #    self.generateHeatmap = GenerateHeatmap(self.output_res, config['inference']['num_parts'])
    #    self.ds = ds
    #    self.index = index

    #def __len__(self):
    #    return len(self.index)

    def __getitem__(self, idx):
        return self.loadImage(idx)

    def loadImage(self, idx):
        image = Image.open(self.img_files_address[idx])
        pose = torch.load(self.pose_files_address[idx])
        w, h = image.size
        image = self.transform(image)
        image = image.numpy()
        image = image[0]
        #image = np.concatenate((image,image, image), axis = 0)
        image = np.stack((image,image, image), axis = 2) 
        pose[0,:] = pose[0,:] + torch.tensor(int((101 - w)/2))
        pose[1,:] = pose[1,:] + torch.tensor(int((101 - h)/2)) 
        
        # Lets send the pose to the output res
        pose *= (self.config['train']['output_res']/ 101)
        temp = torch.zeros(24,2)
        temp[:,0] = pose[0,:]
        temp[:,1] = pose[1,:]
        pose = temp

        ## generate heatmaps on outres
        heatmaps = self.generateHeatmap([pose])
        
        return image.astype(np.float32), heatmaps.astype(np.float32)

    #def preprocess(self, data):
    #    # random hue and saturation
    #    data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV);
    #    delta = (np.random.random() * 2 - 1) * 0.2
    #    data[:, :, 0] = np.mod(data[:,:,0] + (delta * 360 + 360.), 360.)
    #
    #    delta_sature = np.random.random() + 0.5
    #    data[:, :, 1] *= delta_sature
    #    data[:,:, 1] = np.maximum( np.minimum(data[:,:,1], 1), 0 )
    #    data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)
    #
    #    # adjust brightness
    #    delta = (np.random.random() * 2 - 1) * 0.3
    #    data += delta
    #
    #    # adjust contrast
    #    mean = data.mean(axis=2, keepdims=True)
    #    data = (data - mean) * (np.random.random() + 0.5) + mean
    #    data = np.minimum(np.maximum(data, 0), 1)
    #    return data


def init(config):
    batchsize = config['train']['batchsize']
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)
    
    #import ref as ds
    #ds.init()
    #train, valid = ds.setup_val_split()
    #dataset = { key: Dataset(config, ds, data) for key, data in zip( ['train', 'valid'], [train, valid] ) }

    use_data_loader = config['train']['use_data_loader']
    
    #training_data_dir = '../OrthographicFishResnetIOU/data'
    training_data_dir = '../OrthographicFishResnetIOUMultiple/data'

    pose_folder = training_data_dir + '/coor_2d/'
    pose_files = sorted(os.listdir(pose_folder))
    pose_files_add = [pose_folder + file_name for file_name in pose_files]

    im_folder = training_data_dir + '/images/'
    im_files = sorted(os.listdir(im_folder))
    im_files_add = [im_folder + file_name for file_name in im_files]

    
    transform = transforms.Compose([padding(), transforms.Resize((256,256)),transforms.ToTensor(),  transforms.ConvertImageDtype(torch.float)])
    data = ZebrafishDataset(config, im_files_add, pose_files_add, transform=transform) 
    train_size = int(len(data)*0.9)
    val_size = len(data) - train_size
    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
 

    loaders = {}
    loaders['train'] = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=config['train']['num_workers'], pin_memory=False) 
    loaders['valid'] = torch.utils.data.DataLoader(val_data, batch_size=batchsize, shuffle=False, num_workers=config['train']['num_workers'], pin_memory=False)  
    
    #for key in dataset:
    #    loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batchsize, shuffle=True, num_workers=config['train']['num_workers'], pin_memory=False)

    def gen(phase):
        batchsize = config['train']['batchsize']
        batchnum = config['train']['{}_iters'.format(phase)]
        loader = loaders[phase].__iter__()
        for i in range(batchnum):
            try:
                imgs, heatmaps = next(loader)
            except StopIteration:
                # to avoid no data provided by dataloader
                loader = loaders[phase].__iter__()
                imgs, heatmaps = next(loader)
            yield {
                'imgs': imgs, #cropped and augmented
                'heatmaps': heatmaps, #based on keypoints. 0 if not in img for joint
            }


    return lambda key: gen(key)
