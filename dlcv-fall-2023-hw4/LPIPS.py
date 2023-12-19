import torch
import torchvision
import os
import sys

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='mean', normalize=True)
# LPIPS needs the images to be in the [-1, 1] range.

pred = sys.argv[1]
gt = sys.argv[2]

pred_files = [f for f in os.listdir(pred) if f.endswith('.png')]
gt_files = [f for f in os.listdir(gt) if f.endswith('.png')]

pred_img = []
gt_img = []

for file in gt_files:
    pred_img.append(torchvision.io.read_image(f"{pred}/{file}", mode=torchvision.io.ImageReadMode.RGB))
    gt_img.append(torchvision.io.read_image(f"{gt}/{file}", mode=torchvision.io.ImageReadMode.RGB))
    
pred_img = torch.stack(pred_img) / 255
gt_img = torch.stack(gt_img) / 255
print("LPIPS: ", lpips(pred_img, gt_img))