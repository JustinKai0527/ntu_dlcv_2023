import torch
from torch.utils.data import DataLoader, Dataset
import os
import csv
import pandas as pd
import glob
import sys
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import numpy as np


class Test_Dataset(Dataset):
    def __init__(self, folder, filenames, transform=None):
        super(Test_Dataset, self).__init__()
        self.filenames = [os.path.join(folder, i) for i in filenames]
        self.transform = transform
        # print(self.filenames)
        self.len = len(self.filenames)
        
    def __getitem__(self, index):
        
        file_name = self.filenames[index]
        img = Image.open(file_name)
        
        if self.transform:
            img = self.transform(img)
            
        return img, os.path.basename(file_name)

    def __len__(self):
        return self.len

if __name__ == "__main__":
    
    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.6062, 0.5748, 0.5421]),
            std=torch.tensor([0.2397, 0.2408, 0.2455])),
    ])
    
    
    img_csv_file = sys.argv[1]
    img_folder = sys.argv[2]
    output_pth = sys.argv[3]
    
    df = pd.read_csv(img_csv_file)
    id = df['id'].to_list()
    file = df['filename'].to_list()
    
    data = Test_Dataset(img_folder, file, test_tfm)

    test_dataloader = DataLoader(data, batch_size=128, shuffle=False)
    
    backbone = torchvision.models.resnet50()
    
    model = nn.Sequential(
        backbone,
        nn.Dropout(0.25),
        nn.Linear(1000, 65)
    )
    model.load_state_dict(torch.load("hw1_model/SSL_model.pt"))
    
    DEVICE = 'cuda'
    label_pred = []
    model.to(DEVICE)
    model.eval()
    for data, _ in test_dataloader:
        
        data = data.to(DEVICE)
        pred = model(data)
        pred = pred.argmax(dim=1)
        
        label_pred.extend(pred.detach().cpu().numpy())
        
    os.makedirs(os.path.dirname(output_pth), exist_ok=True)

    with open(output_pth, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(('id', 'filename', 'label'))
        for data in zip(id, file, label_pred):
            writer.writerow(data)