import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from DANN_SVHN_model import FeatureExtractor as FE_svhn
from DANN_SVHN_model import LabelPredictor as LP_svhn
from DANN_USPS_model import FeatureExtractor as FE_usps
from DANN_USPS_model import LabelPredictor as LP_usps
import numpy as np
import glob
from PIL import Image
import os
import csv
import sys  

class test_dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.filename = sorted(glob.glob(os.path.join(root, "*.png")))
        self.transform = transform
        self.len = len(self.filename)
    
    def __getitem__(self, index):
        img_fn = self.filename[index]
        img = Image.open(img_fn)
        img = img.convert('RGB')
            
        img = self.transform(img)
            
        return img, os.path.basename(img_fn)

    def __len__(self):
        return self.len
        
        
if __name__ == "__main__":
    
    root = sys.argv[1]
    output_pth = sys.argv[2]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if "svhn" in root:
        test_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4413, 0.4458, 0.4715], std=[0.1169, 0.1206, 0.1042])
        ])
        feature_model = FE_svhn().to(DEVICE)
        pred_model = LP_svhn().to(DEVICE)
        model_ckpt = torch.load("hw2_model/svhn_model.pt")
        feature_model.load_state_dict(model_ckpt["feature_model"])
        pred_model.load_state_dict(model_ckpt["label_pred_model"])
    else:
        test_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2570, 0.2570, 0.2570], std=[0.3372, 0.3372, 0.3372]),
        ])
        feature_model = FE_usps().to(DEVICE)
        pred_model = LP_usps().to(DEVICE)
        model_ckpt = torch.load("hw2_model/usps_model.pt")
        feature_model.load_state_dict(model_ckpt["feature_model"])
        pred_model.load_state_dict(model_ckpt["label_pred_model"])
    
    test = test_dataset(root, test_tfm)
    test_loader = DataLoader(test, batch_size=256, shuffle=False)

    img_file_fn = []
    img_cls_pred = []
    feature_model.eval(), pred_model.eval()
            
    for data, fn in test_loader:
        data = data.to(DEVICE)

        feature = feature_model(data)
        pred = pred_model(feature)
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
        writer.writerow(('image_name', 'label'))
        for data in zip(img_file_fn, img_cls_pred):
            writer.writerow(data)

