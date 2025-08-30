import os
import pdb
import numpy as np
import torch
from tqdm import tqdm

imageSizeY = 602 // 4
imageSizeX = 741 // 4

annotations_path = 'Chie_test_dataset_constant_fish_100k/labels/train/'
output_annotations_path = 'Chie_test_dataset_constant_fish_100k/resnet_labels/train_compressed_4/'
os.makedirs(output_annotations_path, exist_ok=True)
files = os.listdir(annotations_path)
pose_ind = np.array([[5,6],[8,9],[11,12],[14,15],[17,18],[20,21],[23,24],[26,27],[29,30],[32,33],[35,36],[38,39]])
pose_ind = pose_ind.tolist()

for filename in tqdm(files):
    file_path = os.path.join(annotations_path, filename)
    with open(file_path, 'r') as f:
        text = f.readlines()
    if not text:
        continue
    num_fish = len(text)
    if not(num_fish == 4):
        print(filename)
        continue
    pose = torch.zeros(num_fish, 2, 12)

    for fish_id, text_line in enumerate(text):
        text_line = text_line.split(' ')
        pose[fish_id, :, :] = torch.tensor([[float(text_line[i]) * imageSizeX, float(text_line[j]) * imageSizeY] for i,j in pose_ind]).T

    output_file_path = os.path.join(output_annotations_path, f"{filename.split('.')[0]}.pt")
    torch.save(pose, output_file_path)
