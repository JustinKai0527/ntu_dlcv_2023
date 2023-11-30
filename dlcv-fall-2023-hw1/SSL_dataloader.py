import torch
from torch.utils.data import DataLoader, Dataset
import os
import glob
from PIL import Image
import torchvision.transforms as transforms

class MiniDataset(Dataset):
    def __init__(self, root, transform):
        super(MiniDataset, self).__init__()
        self.filenames = []
        self.root = root
        self.transform = transform
    
                
        self.filenames = glob.glob(os.path.join(root, "*"))
        # print(len(self.filenames))     # 38400 
        
    def __getitem__(self, index):
        
        img = self.filenames[index]
        img = Image.open(img)
        if self.transform:
            img = self.transform(img)
        
        
        
        return img
        
    def __len__(self):
        return len(self.filenames)
    

class OfficeDataset(Dataset):
    def __init__(self, root, transform=None):
        super(OfficeDataset, self).__init__()
        self.filenames = []
        self.root = root
        self.transform = transform
        
        for i in range(65):
            
            filenames = sorted(glob.glob(os.path.join(self.root, f"{i}_*.jpg")))
            for fn in filenames:
                self.filenames.append((fn, i))
            # print(len(filenames))
        self.len = len(self.filenames)
        
    def __getitem__(self, index):
        
        img, label = self.filenames[index]
        img = Image.open(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

    def __len__(self):
        return self.len

# if __name__ == "__main__":
#     data = OfficeDataset("hw1_data/p2_data/office/val", transforms.ToTensor())
#    # print(data.__len__())               #406 img   
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
    
#     for i, _ in data:
#         mean += i.mean(dim=(1,2))
#         std += i.std(dim=(1,2))
        
#     mean /= len(data)
#     std /= len(data)
    
#     print(mean, std)
#    # tensor([0.6062, 0.5748, 0.5421]) tensor([0.2397, 0.2408, 0.2455])
        
    
#     data = MiniDataset("hw1_data/p2_data/mini/train", transforms.ToTensor())
    
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
    
#     for i in data:
#         mean += i.mean(dim=(1,2))
#         std += i.std(dim=(1,2))
        
#     mean /= len(data)
#     std /= len(data)
    
#     print(mean, std)
    # tensor([0.4705, 0.4495, 0.4037]) tensor([0.2170, 0.2149, 0.2145])