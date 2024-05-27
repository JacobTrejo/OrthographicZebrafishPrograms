# This script trains a network model to predict larval poses, given synthetic images

import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from CustomDataset import CustomImageDataset
#from ResNet_Blocks_3D_four_blocks import resnet18
from ResNet_Blocks_3D_six_blocks import resnet18
import time
from multiprocessing import Pool
import os
import pdb

torch.manual_seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs',default=10, type=int, help='number of epochs to train the VAE for')
parser.add_argument('-o','--output_dir', default="outputs/220704_old_model", type=str, help='path to store output images and plots')
parser.add_argument('-t','--training_data_dir', default="../../../training_data_2D_230511", type=str, help='path to store output images and plots')

args = vars(parser.parse_args())
imageSizeX = 101
imageSizeY = 101

epochs = args['epochs']
output_dir = args['output_dir']
training_data_dir = args['training_data_dir']

lr = 0.001
#lr = 0.01

if (not os.path.isdir(output_dir)):
    os.mkdir(output_dir)
    print('Creating new directory to store output images')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(1, 24, activation='leaky_relu').to(device)

n_cuda = torch.cuda.device_count()
if (torch.cuda.is_available()):
    print(str(n_cuda) + 'GPUs are available!')
    nworkers = n_cuda*12
    pftch_factor = 2
else: 
    print('Cuda is not available. Training without GPUs. This might take long')
    nworkers = 4
    pftch_factor = 2
batch_size = 512*n_cuda

if torch.cuda.device_count() > 1:
  print("Using " + str(n_cuda) + " GPUs!")
  model = nn.DataParallel(model)

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

transform = transforms.Compose([padding(), transforms.ToTensor(),  transforms.ConvertImageDtype(torch.float)])
pose_folder = training_data_dir + '/coor_2d/'
pose_files = sorted(os.listdir(pose_folder))
pose_files_add = [pose_folder + file_name for file_name in pose_files]

im_folder = training_data_dir + '/images/'
im_files = sorted(os.listdir(im_folder))
im_files_add = [im_folder + file_name for file_name in im_files]

data = CustomImageDataset(im_files_add, pose_files_add, transform=transform)
train_size = int(len(data)*0.9)
val_size = len(data) - train_size
train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
print(len(data))

train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True,num_workers=nworkers,prefetch_factor=pftch_factor,persistent_workers=True)
val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=False,num_workers=nworkers,prefetch_factor=pftch_factor,persistent_workers=True)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='sum')
criterion2 = nn.MSELoss()

def final_loss(mse_loss, mu, logvar):
    MSE = mse_loss
    KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    running_pose_loss = 0.0
    #for i, data in enumerate(dataloader):
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        im, pose_data = data
        im = im.to(device)
        pose_data = pose_data 
        pose_data = pose_data.to(device)
        optimizer.zero_grad()
        pose_recon = model(im)
        #print('pose_recon shape: ', pose_recon.shape)
        #print('pose data shape: ', pose_data.shape)
        
        # Symetrizing the fish with respect to each other
        fish1GT = pose_data[:,:,:12].clone()
        fish2GT = pose_data[:,:,12:].clone()

        fish1 = pose_recon[:,:,:12].clone()
        fish2 = pose_recon[:,:,12:].clone()
        fish1Good = fish1.clone().detach()
        fish2Good = fish2.clone().detach()
        
        backLossReg  = torch.sum(torch.sum((fish1GT[:,:,:10] - fish1[:,:,:10])**2, axis = 1)**.5, axis = -1)
        backLossFlip = torch.sum(torch.sum((fish1GT[:,:,:10] - fish2[:,:,:10])**2, axis = 1)**.5, axis = -1)
        flipMask = backLossFlip < backLossReg
        
        fish1Good[flipMask] = fish2[flipMask]
        #fish2Good[~flipMask] = fish1[~flipMask]
        fish2Good[flipMask] = fish1[flipMask]

        # Now let symetrise the eye loss
        fish1 = fish1Good.clone().detach()
        fish2 = fish2Good.clone().detach()
        
        fish1[:,:,10:] = torch.flip(fish1[:,:,10:], [-1])
        fish2[:,:,10:] = torch.flip(fish2[:,:,10:], [-1])
        
        backLossReg  = torch.sum(torch.sum((fish1GT - fish1Good)**2, axis = 1)**.5, axis = -1)
        backLossFlip = torch.sum(torch.sum((fish1GT - fish1)**2, axis = 1)**.5, axis = -1)
        flipMask = backLossFlip < backLossReg
        fish1Good[flipMask] = fish1[flipMask]
        
        backLossReg  = torch.mean(torch.sum((fish2GT - fish2Good)**2, axis = 1)**.5, axis = -1)
        backLossFlip = torch.mean(torch.sum((fish2GT - fish2)**2, axis = 1)**.5, axis = -1)
        flipMask = backLossFlip < backLossReg
        fish2Good[flipMask] = fish2[flipMask]
        
        pose_recon[:,:,:12] = fish1Good
        pose_recon[:,:,12:] = fish2Good

        pose_loss = criterion(pose_recon, pose_data)
        loss = pose_loss 
        running_loss += loss.item()
        running_pose_loss += pose_loss.item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss/len(dataloader.dataset)
    train_pose_loss = running_pose_loss/len(dataloader.dataset)
    return train_loss, train_pose_loss

def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    running_pose_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            im, pose_data = data
            im = im.to(device)
            pose_data = pose_data.to(device)
            pose_recon = model(im)
            
            fish1GT = pose_data[:,:,:12].clone().detach()
            fish2GT = pose_data[:,:,12:].clone().detach()

            fish1 = pose_recon[:,:,:12].clone().detach()
            fish2 = pose_recon[:,:,12:].clone().detach()
            fish1Good = fish1.clone().detach()
            fish2Good = fish2.clone().detach()

            backLossReg  = torch.sum(torch.sum((fish1GT[:,:,:10] - fish1[:,:,:10])**2, axis = 1)**.5, axis = -1)
            backLossFlip = torch.sum(torch.sum((fish1GT[:,:,:10] - fish2[:,:,:10])**2, axis = 1)**.5, axis = -1)
            flipMask = backLossFlip < backLossReg

            fish1Good[flipMask] = fish2[flipMask]
            #fish2Good[~flipMask] = fish1[~flipMask]
            fish2Good[flipMask] = fish1[flipMask]
            
            # Now let symetrise the eye loss
            fish1 = fish1Good.clone().detach()
            fish2 = fish2Good.clone().detach()

            fish1[:,:,10:] = torch.flip(fish1[:,:,10:], [-1])
            fish2[:,:,10:] = torch.flip(fish2[:,:,10:], [-1])

            backLossReg  = torch.sum(torch.sum((fish1GT - fish1Good)**2, axis = 1)**.5, axis = -1)
            backLossFlip = torch.sum(torch.sum((fish1GT - fish1)**2, axis = 1)**.5, axis = -1)
            flipMask = backLossFlip < backLossReg
            fish1Good[flipMask] = fish1[flipMask]

            backLossReg  = torch.mean(torch.sum((fish2GT - fish2Good)**2, axis = 1)**.5, axis = -1)
            backLossFlip = torch.mean(torch.sum((fish2GT - fish2)**2, axis = 1)**.5, axis = -1)
            flipMask = backLossFlip < backLossReg
            fish2Good[flipMask] = fish2[flipMask]

            pose_recon[:,:,:12] = fish1Good
            pose_recon[:,:,12:] = fish2Good
            
            pose_loss = criterion(pose_recon, pose_data) 
            loss = pose_loss 
            running_loss += loss.item()
            running_pose_loss += pose_loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 8
                im = im[:,0,:,:]
                images = im.view(batch_size, 1,imageSizeY,imageSizeX)[:8]
                _, axs = plt.subplots(nrows=2, ncols=8)
                images = torch.squeeze(images)

                # Overlay pose
                for m in range(0,8):
                    axs[1,m].imshow(images[m,:,:].cpu(), cmap='gray')
                    axs[1,m].scatter(pose_recon[m,0,:].cpu(), pose_recon[m,1,:].cpu(), s=0.07, c='green', alpha=0.6)
                    axs[1,m].axis('off')

                for m in range(0,8):
                    axs[0,m].imshow(images[m,:,:].cpu(), cmap='gray')
                    #axs[0,m].scatter(pose_data[m,0,0:12].cpu(), pose_data[m,1,0:12].cpu(), s=0.07, c='red', alpha=0.6)
                    axs[0,m].scatter(pose_data[m,0,:].cpu(), pose_data[m,1,:].cpu(), s=0.07, c='red', alpha=0.6)
                    axs[0,m].axis('off')


                plt.savefig(output_dir + "/epoch_" + str(epoch) + ".svg")
                plt.close()
    val_loss = running_loss/len(dataloader.dataset)
    val_pose_loss = running_pose_loss/len(dataloader.dataset)
    return val_loss, val_pose_loss

train_loss = []
val_loss = []
train_pose_loss_array = []
val_pose_loss_array = []

best_loss = 5000
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}",flush=True)
    train_epoch_loss, train_pose_loss = fit(model, train_loader)
    val_epoch_loss, val_pose_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    if (val_epoch_loss < best_loss):
        torch.save(model.state_dict(), 'resnet_pose_best_python_230608_six_blocks.pt')
        best_loss = val_epoch_loss
        print('Saving new model. New best_loss = ', str(best_loss))
    train_pose_loss_array.append(train_pose_loss)
    val_pose_loss_array.append(val_pose_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}",flush=True)
    print(f"Val Loss: {val_epoch_loss:.4f}",flush=True)


plt.plot(train_loss[20:], color='green')
plt.plot(val_loss[20:], color='red')
plt.plot(train_pose_loss_array[20:], linestyle='--', color='green')
plt.plot(val_pose_loss_array[20:], linestyle='--', color='red')
plt.savefig(output_dir + "/loss_truncated.png")

plt.plot(train_loss, color='green')
plt.plot(val_loss, color='red')
plt.savefig(output_dir + "/loss.png")

