import torch 
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from classification_model import VGG13
import numpy as np
import glob
from PIL import Image
import os
import sys
import imageio

class SegmentationDataset(Dataset):
    def __init__(self, root):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.img_file = None
        self.mask_file = None
        
        self.img_file = sorted(glob.glob(os.path.join(self.root, "*_sat.jpg")))
    
    def __getitem__(self, index):
        fn = self.img_file[index]
        img = Image.open(self.img_file[index])
        
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.4085, 0.3785, 0.2809], std=[0.1155, 0.0895, 0.0772])(img)
        
        return img, os.path.basename(fn)     # get the base name
        
    def __len__(self):
        return len(self.img_file)
    
def get_mask_img(fn, mask_pred, output_pth):
    

    for pred, name in zip(mask_pred, fn):

        pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
        pred_img[np.where(pred == 0)] = [0, 255, 255]
        pred_img[np.where(pred == 1)] = [255, 255, 0]
        pred_img[np.where(pred == 2)] = [255, 0, 255]
        pred_img[np.where(pred == 3)] = [0, 255, 0]
        pred_img[np.where(pred == 4)] = [0, 0, 255]
        pred_img[np.where(pred == 5)] = [255, 255, 255]
        pred_img[np.where(pred == 6)] = [0, 0, 0]

        imageio.imwrite(os.path.join(output_pth, name.replace('_sat.jpg', '_mask.png')), pred_img)
        
if __name__ == "__main__":
    
    root = sys.argv[1]
    output_pth = sys.argv[2]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test = SegmentationDataset(root)
    test_loader = DataLoader(test, batch_size=4, shuffle=False)

    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, 7, 1, 1)
    model.aux_classifier[4] = nn.Conv2d(256, 7, 1, 1)
    model.load_state_dict(torch.load("hw1_model/seg_model.pt"))
    model.to(DEVICE)

    model.eval()
    img_file_fn = []
    img_mask_pred = []
    
    for data, imgae_fn in test_loader:
        data = data.to(DEVICE)
        
        pred = model(data)['out']
        pred = pred.argmax(dim=1)
        
        img_mask_pred.extend(pred.detach().cpu().numpy())
        img_file_fn.extend(imgae_fn)
    
    img_file_fn = np.array(img_file_fn, dtype=str)
    img_mask_pred = np.array(img_mask_pred, dtype=np.uint8)
    
    os.makedirs(output_pth, exist_ok=True)
    get_mask_img(img_file_fn, img_mask_pred, output_pth)
        

