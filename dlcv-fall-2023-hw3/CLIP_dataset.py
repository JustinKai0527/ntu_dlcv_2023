import torch
from torch.utils.data import Dataset, DataLoader 
import torchvision
import numpy as np
import glob
import os
from PIL import Image

# bash hw3_1.sh $1 $2 $3
# $1: path to the folder containing test images (images are named xxxx.png, where xxxx could be any string)
# $2: path to the id2label.json
# $3: path of the output csv file (e.g. output_p1/pred.csv)

class CLIP_dataset(Dataset):
    def __init__(self, root, preprocess):
        
        self.root = root
        self.filenames = sorted(glob.glob(os.path.join(root, "*.png")))
        self.preprocess = preprocess
        
        self.len = len(self.filenames)
        
    def __getitem__(self, index):
        
        image_fn = self.filenames[index]
        label = int(os.path.basename(image_fn).split("_")[0])
        img = Image.open(image_fn)

        img = self.preprocess(img)

        return img, label
    
    def __len__(self):
        return self.len
    
# str = "01_450.png"
# print(str.split("_")[0])       # 01
