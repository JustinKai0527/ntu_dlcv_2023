from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
from PIL import Image
import json
import matplotlib.pyplot as plt
from tokenizer import BPETokenizer
import torchvision.transforms as transforms
from tqdm import tqdm

class PEFT_dataset(Dataset):
    def __init__(self, root, transform=None, json_file_pth=None, max_len=75):
        super(PEFT_dataset, self).__init__()
        
        self.root = root
        self.encoding = BPETokenizer("encoder.json", "vocab.bpe")
        self.info = None
        if json_file_pth:
            with open(json_file_pth, 'r') as file:
                info = json.load(file)

            self.info = info
        
        self.tfm = transform
        self.max_len = max_len
        self.len = len(self.info["annotations"])
        
    def __getitem__(self, index):
        
        captions_info = self.info["annotations"][index]
        caption = captions_info["caption"]
        image_id = captions_info["image_id"]
        images_info = self.info["images"]
        # print(images_info)
        img_file_name = [item["file_name"] for item in images_info if item["id"] == image_id]

        img = Image.open(os.path.join(self.root, img_file_name[0])).convert('RGB')
        
        if self.tfm:
            img = self.tfm(img)
            
        token = np.array(self.encoding.encode(caption))
        tmp = [*token, 50256]
        # if(self.max_len - len(tmp)) < 0:
        #     print(self.max_len - len(tmp))
        #     print(tmp)
        gt = np.pad(tmp, pad_width=(0, self.max_len - len(tmp)), mode='constant', constant_values=(-100, -100))
        # # # print(gt)
        token = np.pad(token, pad_width=(1, self.max_len - len(token) - 1), mode='constant', constant_values=(50256, 50256))

        return img, token, gt
    
    def __len__(self):
        return self.len
    
    

        
# with open("hw3_data/p2_data/train.json", 'r') as file:
#     info = json.load(file)
# # # print(len(info["annotations"]))       
# tfm = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
# ])
# dataset = PEFT_dataset("hw3_data/p2_data/images/train", tfm, "hw3_data/p2_data/train.json")
# dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

# for i, (img, token, gt) in enumerate(dataset):

    # print(token, gt, len(token) == len(gt))
    # if i == 3:
    #     break
    # if len(token) != len(gt):
    #     print(token)
# print(dataset.__len__())
# encoding = BPETokenizer("encoder.json", "vocab.bpe")
# max_len = 0
# max_caption = None
# for img, token in tqdm(dataset):
    # print(img, caption)
    # img = np.array(img)
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()
#     token = encoding.encode(caption)
#     if len(token) > max_len:
#         max_caption = token
#     max_len = len(token) if len(token) > max_len else max_len

# print(max_len, max_caption)
# 55 visible through a windshield:  in the distance, sidewalks, lined with snow, utility poles, retail outposts, 
# and  a few  approaching vehicles, in the foreground, a crosswalk with a turning truck at one side and two large vehicles directly past it.