import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transform
import numpy as np
import pandas as pd
import os
from PIL import Image


class Mnistm_dataset(Dataset):
    def __init__(self, root, filename, label, transform=None, domain=False):
        
        super(Mnistm_dataset, self).__init__()
        index = np.argsort(filename)
        self.root = root
        self.labels = label[index]
        filename = filename[index]
        # print(len(self.labels), len(filename))             # 44800
        self.filenames = [os.path.join(root, fn) for fn in filename]
        self.transform = transform
        self.domain = domain
        self.len = len(self.filenames)
        
    def __getitem__(self, index):
        image_fn = self.filenames[index]
        label = self.labels[index]
        img = Image.open(image_fn)
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transform.ToTensor()(img)
            
        if self.domain:
            return img, 0, label
        
        return img, label
    
    def __len__(self):
        return self.len
        

        
class SVHN_dataset(Dataset):
    def __init__(self, root, filename, label, transform=None, domain=False):
        super(SVHN_dataset, self).__init__()
        
        index = np.argsort(label)
        filename = filename[index]
        self.labels = label[index]
        self.root = root
        self.domain = domain
        self.filenams = [os.path.join(root, fn) for fn in filename]
        
        # print(len(self.labels), len(filename))             # 63544
        self.transform = transform
        self.len = len(self.filenams)
        
    def __getitem__(self, index):
        
        image_fn = self.filenams[index]
        img = Image.open(image_fn)
        label = self.labels[index]
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transform.ToTensor()(img)
            
        if self.domain:
            return img, 1, label
        
        return img, label
        
    def __len__(self):
        
        return self.len
    
    
class USPS_dataset(Dataset):
    def __init__(self, root, filename, label, transform=None, domain=False):
        super(USPS_dataset, self).__init__()
        
        self.root = root
        index = np.argsort(label)
        self.labels = label[index]
        filename = filename[index]
        self.filenames = [os.path.join(root, fn) for fn in filename]
        self.transform = transform
        self.domain = domain
        self.len = len(self.filenames)
        
    def __getitem__(self, index):
        
        image_fn = self.filenames[index]
        img = Image.open(image_fn)
        img = img.convert('RGB')
        label = self.labels[index]
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transform.ToTensor()(img)
        
        if self.domain:
            return img, 1, label       # due to the img shape (1, 28, 28)
        
        return img, label        
        
    def __len__(self):
        return self.len
    
    # mnistm (3, 28, 28)
    # svhn (3, 28, 28)
    # usps (1, 28, 28)
    
    
# if __name__ == "__main__":
    
#     pd1 = pd.read_csv("hw2_data/digits/usps/train.csv")
#     filename = pd1["image_name"].to_numpy()
#     labels = pd1["label"].to_numpy()
#     root = "hw2_data/digits/usps/data"
#     dataset1 = USPS_dataset(root, filename, labels, domain=False)

#     print(dataset1.__len__())
    
#     # dataloader = DataLoader(dataset1, batch_size=256, shuffle=False)
#     mean, std = torch.zeros(3), torch.zeros(3)
    
#     for data, label in dataset1:
#         print(data.shape)
#         mean += data.mean(dim=(1, 2))
#         std += data.std(dim=(1, 2))
    
#     print(mean / len(dataset1), std / len(dataset1))
    # mnistm tensor([0.4631, 0.4666, 0.4195]) tensor([0.1979, 0.1845, 0.2083])
    # svhn tensor([0.4413, 0.4458, 0.4715]) tensor([0.1169, 0.1206, 0.1042])
    # usps tensor([0.2570, 0.2570, 0.2570]) tensor([0.3372, 0.3372, 0.3372])
    
    
    