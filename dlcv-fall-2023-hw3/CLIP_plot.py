import torch
from torch.utils.data import DataLoader 
import clip
from PIL import Image
from CLIP_dataset import CLIP_dataset
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# https://github.com/openai/CLIP
# bash hw3_1.sh $1 $2 $3
# $1: path to the folder containing test images (images are named xxxx.png, where xxxx could be any string)
# $2: path to the id2label.json
# $3: path of the output csv file (e.g. output_p1/pred.csv)

# data_pth = sys.argv[1]
# json_file_pth = sys.argv[2]
# output_csv = sys.argv[3]
# print(clip.available_models())               # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
model, preprocess = clip.load("ViT-B/16", device=DEVICE)

# os.makedirs(output_csv, exist_ok=True)

json_file_pth = "hw3_data/p1_data/id2label.json"
with open(json_file_pth, 'r') as file:
    cls = json.load(file)

cls_num = len(cls)
cls_name = [cls[str(i)] for i in range(cls_num)]
img_pth = ["8_480.png", "45_486.png", "36_454.png"]
img = []
top5_prob = []
top5_label = []

text_token = clip.tokenize([f"a photo of a {object}"for object in cls_name]).to(DEVICE)

for pth in img_pth:
    
    img.append(np.array(Image.open(pth)))
    image = preprocess(Image.open(pth)).reshape(1, 3, 224, 224).to(DEVICE)
    model.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        # print(image.shape, text_token.shape)
        
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_token)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (50 * image_features @ text_features.T).softmax(dim=-1)
        # print(probs.sum(dim=1))
        top5_prob.append(similarity.topk(5)[0].cpu().numpy())
        top5_label.append(similarity.topk(5)[1].cpu().numpy())

top5_prob = np.array(top5_prob).reshape(-1, 5)
top5_label = np.array(top5_label).reshape(-1, 5)
# print(top5_prob.shape, top5_label.shape)
print(top5_prob)
num_images = len(img)
img_size = img[0].shape[:2]  # Assuming all images have the same size

fig, axes = plt.subplots(1, num_images * 2, figsize=(36, 4))

rgb = np.array([0.1, 0.7, 0.1]).reshape(1, 3).repeat(repeats=[5], axis=0)

for i, image in enumerate(img):
    axes[2 * i].imshow(image)
    axes[2 * i].axis("off")

    color = np.concatenate((rgb, top5_prob[i].reshape(-1, 1) + 0.2), axis=1)
    # print(color.shape)
    
    
    y = np.arange(top5_prob.shape[-1])
    axes[2 * i + 1].grid()
    axes[2 * i + 1].barh(y, top5_prob[i], color=color)
    axes[2 * i + 1].invert_yaxis()
    axes[2 * i + 1].set_axisbelow(True)
    for idx, item in enumerate(top5_label[i]):
        axes[2 * i + 1].text(x=0, y=idx, s=f"a photo of {cls_name[item]}", va='center', ha='left')
    axes[2 * i + 1].set_xlabel("probability")

    axes[2 * i + 1].get_yaxis().set_visible(False)
# Adjust the aspect ratio of the imshow to maintain the same image size
for ax in axes[::2]:
    ax.set_aspect('auto')

plt.subplots_adjust(wspace=0)
plt.show()

