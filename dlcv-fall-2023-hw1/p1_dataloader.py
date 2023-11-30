import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# https://colab.research.google.com/drive/1MPPkhLB6gp875SThfAWTMeRVHFOV51IE?usp=sharing#scrollTo=C4M-ON6BCttO
class p1_dataset(Dataset):
    def __init__(self, root, transform=None):
        
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        
        # 50 cls
        for i in range(50):
            
            filenames = sorted(glob.glob(os.path.join(root, str(i) + "_*.png")))    # must have "_" or you may got wrong...
            # print(len(filenames))
            for fn in filenames:
                self.filenames.append((fn, i))   # (filename, label)
                
        self.len = len(self.filenames)
        
    def __getitem__(self, index):
        image_fn, label = self.filenames[index]
        img = Image.open(image_fn)   # <class 'PIL.PngImagePlugin.PngImageFile'>

        if self.transform != None:
            img = self.transform(img)
        
        return img, label
        
    def __len__(self):
        return self.len
    

# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#     return
    
# if __name__ == "__main__":
    
#     train = p1_dataset(root="hw1_data/p1_data/train_50", transform=transforms.ToTensor())
#     valid = p1_dataset(root="hw1_data/p1_data/val_50", transform=None)
    
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     for img, _ in train:
#         mean += torch.mean(img, dim=(1, 2))
#         std += torch.std(img, dim=(1, 2))
        
    # print(mean / len(train))     tensor([0.5077, 0.4813, 0.4312])
    # print(std / len(train))      tensor([0.2000, 0.1986, 0.2034])
    # print(img.to('cuda'), label)
    # print(train.__len__())    # 22500
    # print(valid.__len__())   # 2500
    
    # train_loader = DataLoader(train, batch_size=128, shuffle=False, num_workers=4)
    # train_loader = iter(train_loader)
    # images, labels = next(train_loader)
    # print(images.shape, labels.shape)
    
    # show images
    # imshow(torchvision.utils.make_grid(images))
    