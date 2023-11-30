import torch
from torch.utils.data import Dataset, DataLoader 
import clip
from PIL import Image
import json
import sys
import os
import glob
import csv
import numpy as np

# https://github.com/openai/CLIP
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
        img = Image.open(image_fn)
        img = self.preprocess(img)

        return img, os.path.basename(image_fn)
    
    def __len__(self):
        return self.len

data_pth = sys.argv[1]
json_file_pth = sys.argv[2]
output_csv = sys.argv[3]
# print(clip.available_models())               # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
model, preprocess = clip.load("ViT-B/16", device=DEVICE)

dataset = CLIP_dataset(data_pth, preprocess)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

with open(json_file_pth, 'r') as file:
    cls = json.load(file)

cls_num = len(cls)
cls_name = [cls[str(i)] for i in range(cls_num)]

with torch.no_grad():
    text_token = clip.tokenize([f"a photo of a {object}"for object in cls_name]).to(DEVICE)

    model.eval()
    img_file_fn = []
    img_cls_pred = []
    
    for img, file_name in dataloader:
        
        img = img.to(DEVICE)
        
        image_features = model.encode_image(img)
        text_features = model.encode_text(text_token)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        
        pred = similarity.argmax(dim=1)
        img_file_fn.extend(file_name)
        img_cls_pred.extend(pred.detach().cpu().numpy())
    
    img_file_fn = np.array(img_file_fn, dtype=str)
    img_cls_pred = np.array(img_cls_pred, dtype=np.uint8)
    
    index = np.argsort(img_file_fn)
    img_file_fn = img_file_fn[index]
    img_cls_pred = img_cls_pred[index]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(('filename', 'label'))
        for data in zip(img_file_fn, img_cls_pred):
            writer.writerow(data)
