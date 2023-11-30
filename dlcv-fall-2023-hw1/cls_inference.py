import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from PIL import Image
import os
import csv
import sys

class img_dataset(Dataset):
    def __init__(self, root):
        
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        
        mean = torch.tensor([0.5077, 0.4813, 0.4312])
        std = torch.tensor([0.2000, 0.1986, 0.2034])
    
        self.tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        self.filenames = glob.glob(os.path.join(root, "*.png"))
        self.len = len(self.filenames)
        
    def __getitem__(self, index):
        image_fn = self.filenames[index]
        img = Image.open(image_fn)   # <class 'PIL.PngImagePlugin.PngImageFile'>

        if self.tfm != None:
            img = self.tfm(img)
        return img, os.path.basename(image_fn)
        
    def __len__(self):
        return self.len
    

if __name__ == "__main__":
    
    root = sys.argv[1]
    # print(root)
    output_pth = sys.argv[2]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test = img_dataset(root)
    test_loader = DataLoader(test, batch_size=8, shuffle=False)
    
    model = torchvision.models.resnet152()
    model.fc = nn.Linear(2048, 50)
    model.load_state_dict(torch.load("hw1_model/cls_model.pt")["model_state_dict"])
    model.to(DEVICE)
    
    model.eval()
    img_file_fn = []
    img_cls_pred = []
    
    for data, fn in test_loader:
        data = data.to(DEVICE)

        pred = model(data)
        pred = pred.argmax(dim=1)
        
        img_file_fn.extend(fn)
        img_cls_pred.extend(pred.detach().cpu().numpy())
    
    img_file_fn = np.array(img_file_fn, dtype=str)
    img_cls_pred = np.array(img_cls_pred, dtype=np.uint8)
    
    index = np.argsort(img_file_fn)
    img_file_fn = img_file_fn[index]
    img_cls_pred = img_cls_pred[index]
    
    os.makedirs(os.path.dirname(output_pth), exist_ok=True)
    
    with open(output_pth, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(('filename', 'label'))
        for data in zip(img_file_fn, img_cls_pred):
            writer.writerow(data)

