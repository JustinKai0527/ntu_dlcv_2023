from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import copy
import random

class SegmentationDataset(Dataset):
    def __init__(self, root):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.img_file = None
        self.mask_file = None
        # self.transform = [transforms.ToTensor(), transforms.RandomHorizontalFlip(1), transforms.RandomVerticalFlip(1), transforms.ToTensor()]
        # self.training = training
        
        # get the img, mask file
        self.img_file = sorted(glob.glob(os.path.join(self.root, "*_sat.jpg")))
        self.mask_file = sorted(glob.glob(os.path.join(self.root, "*_mask.png")))
        
        
        # got img, mask 2000
        # print(len(self.img_file))
        # print(len(self.mask_file))
    
    def __getitem__(self, index):
        img = Image.open(self.img_file[index])
        mask = Image.open(self.mask_file[index])
         
        # if self.training:
        #     tfm = random.choice(self.transform)
        #     img = tfm(img)
        #     if str(tfm) != "ToTensor()":     # if you do the transforms.ToTensor() != tfm: you got two instasnce so it always same
        #         img = transforms.ToTensor()(img)
        #         mask = tfm(mask)
        # else:
        #     img = transforms.ToTensor()(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.4085, 0.3785, 0.2809], std=[0.1155, 0.0895, 0.0772])(img)
        
        
        mask = np.array(mask)
        # print(mask.shape)      shape (512, 512, 3)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        # we have to got the index of the mask
        index = copy.deepcopy(mask)
        mask[index == 3] = 0  # (Cyan: 011) Urban land 
        mask[index == 6] = 1  # (Yellow: 110) Agriculture land 
        mask[index == 5] = 2  # (Purple: 101) Rangeland 
        mask[index == 2] = 3  # (Green: 010) Forest land 
        mask[index == 1] = 4  # (Blue: 001) Water 
        mask[index == 7] = 5  # (White: 111) Barren land 
        mask[index == 0] = 6  # (Black: 000) Unknown 
        
        # turn back to tensor
        mask = torch.tensor(mask)
        
        return img, mask
        
    def __len__(self):
        return len(self.img_file)
    
def get_mask(mask):
    
    mask = mask.cpu().numpy()
    index = copy.deepcopy(mask)
    mask = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2])
    mask_img = mask.repeat(3, axis=1).transpose((0, 2, 3, 1))
    
    mask_img[np.where(index==0)] = np.array([0, 255, 255])
    mask_img[np.where(index==1)] = np.array([255, 255, 0])
    mask_img[np.where(index==2)] = np.array([255, 0, 255])
    mask_img[np.where(index==3)] = np.array([0, 255, 0])
    mask_img[np.where(index==4)] = np.array([0, 0, 255])
    mask_img[np.where(index==5)] = np.array([255, 255, 255])
    mask_img[np.where(index==6)] = np.array([0, 0, 0])
    mask_img = torch.tensor(mask_img.transpose((0, 3, 1, 2)))
    mask_img = torchvision.utils.make_grid(mask_img)
    imshow(mask_img)
    
def imshow(img):
    img = torch.squeeze(img)
    img = np.array(img)
    img = img.transpose((1, 2, 0))
    # plt.imshow(): (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
    plt.imshow(img)
    plt.show()

# if __name__ == "__main__":

#     train = SegmentationDataset("hw1_data/p3_data/validation")
#     train_loader = DataLoader(train, batch_size=32, shuffle=False, num_workers=4)

#     # # print(imshow(img[0]))

#     for img, mask in train_loader:
#         imshow(torchvision.utils.make_grid(img))
#         get_mask(torchvision.utils.make_grid(mask))
    # img_mean, img_std = 0, 0
    # for img, _ in train:
    #     img_mean += img.mean(dim=(1,2))
    #     img_std += img.std(dim=(1,2))
    # print(img_mean / len(train))
    # print(img_std / len(train))
    # tensor([0.4085, 0.3785, 0.2809])
    # tensor([0.1155, 0.0895, 0.0772])