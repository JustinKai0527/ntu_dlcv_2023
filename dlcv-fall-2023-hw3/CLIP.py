import torch
from torch.utils.data import DataLoader 
import clip
from PIL import Image
from CLIP_dataset import CLIP_dataset
import json
import sys
import os

# https://github.com/openai/CLIP
# bash hw3_1.sh $1 $2 $3
# $1: path to the folder containing test images (images are named xxxx.png, where xxxx could be any string)
# $2: path to the id2label.json
# $3: path of the output csv file (e.g. output_p1/pred.csv)

data_pth = sys.argv[1]
json_file_pth = sys.argv[2]
output_csv = sys.argv[3]
# print(clip.available_models())               # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
model, preprocess = clip.load("ViT-B/16", device=DEVICE)

dataset = CLIP_dataset(data_pth, preprocess)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

# os.makedirs(output_csv, exist_ok=True)

with open(json_file_pth, 'r') as file:
    cls = json.load(file)

cls_num = len(cls)
cls_name = [cls[str(i)] for i in range(cls_num)]


with torch.no_grad():
    text_token = clip.tokenize([f"a photo of a {object}"for object in cls_name]).to(DEVICE)
    # text_token = clip.tokenize([f"This a photo of {object}"for object in cls_name]).to(DEVICE)
    # text_token = clip.tokenize([f"This is not a photo of {object}"for object in cls_name]).to(DEVICE)
    # text_token = clip.tokenize([f"None {object}, no score"for object in cls_name]).to(DEVICE)

    acc = []
    model.eval()
    for img, label in dataloader:
        
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        
        # logits_image, _ = model(img, text_token)
        image_features = model.encode_image(img)
        text_features = model.encode_text(text_token)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        
        # print(logits_image.shape, logits_text.shape)              # torch.Size([128, 50]) torch.Size([50, 128])
        # logits_image = (logits_image).softmax(dim=1)
        pred = similarity.argmax(dim=1)
        # print(pred.shape)
        # print(pred, label)
        
        acc.append((pred == label).float().mean().item())

    print(sum(acc)/ len(acc))


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("0_450.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a bicycle", "a dog", "a flower"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
