import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tokenizer import BPETokenizer
import timm
from visualize_decoder import Decoder, Config
from PIL import Image
import json
import sys
import os
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import re

class test_dataset(Dataset):
    def __init__(self, root, tfm=None):
        
        self.root = root
        self.img_file = glob.glob(os.path.join(root, "*"))
        self.tfm = tfm
        self.len = len(self.img_file)
        
    def __getitem__(self, index):
        
        img = Image.open(self.img_file[index]).convert('RGB')
        
        if self.tfm:
            img = self.tfm(img)

        return img, self.img_file[index]
    
    def __len__(self):
        return self.len
# bash hw3_2.sh $1 $2 $3
# $1: path to the folder containing test images (e.g. hw3/p2_data/images/test/)
# $2: path to the output json file (e.g. hw3/output_p2/pred.json) 
# $3: path to the decoder weights (e.g. hw3/p2_data/decoder_model.bin)
# (This means that you don’t need to upload decoder_model.bin)

if __name__ == "__main__":
    
    image_pth = sys.argv[1]
    output_json_file = sys.argv[2]
    decoder_weight_pth = sys.argv[3]
    
    test_tfm = transforms.Compose([
        transforms.Resize((378, 378)),
        transforms.ToTensor(),
    ])
    
    dataset = test_dataset(image_pth, test_tfm)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    encoder = timm.create_model('vit_huge_patch14_clip_378', pretrained=True)
    cfg = Config(decoder_weight_pth)
    decoder = Decoder(cfg)
    decoder.load_state_dict(torch.load("model/model_3.pt"), strict=False)
    encoding = BPETokenizer("encoder.json", "vocab.bpe")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.to(DEVICE)
    decoder.to(DEVICE)
    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        result_dict = {}
        for index, (img, file) in tqdm(enumerate(dataloader)):
            
            img = img.to(DEVICE)
            enc_feat = encoder.forward_features(img).narrow(1, 1, 729)
            token = torch.tensor([50256]).reshape(-1, 1).to(DEVICE)
            attention_map = []
            while True:
                
                pred_token, att_weight = decoder(token, enc_feat)
                pred_token = nn.Softmax(dim=2)(pred_token)                 # make sure the prob is non-negative
                # gready search
                # print(torch.topk(pred_token, 3, dim=2))
                pred_token = pred_token.argmax(dim=2)
                token = torch.cat([token, pred_token[0, -1].reshape(-1, 1)], dim=1)
                attention_map.append(att_weight[:, -1, :])
                # top-K sampling
                # k = 5
                # top_k_tokens = torch.argsort(pred_token, dim=2, descending=True)[0, -1, :k]           # N, T, E
                # top_k_probs = pred_token[0, -1, top_k_tokens]  # the weight feed into the multinomial doesn't have to sum to 1
                # # print(top_k_probs.shape, top_k_tokens.shape)
                # sample_token_index = torch.multinomial(top_k_probs, 1)
                # sample_token = top_k_tokens[sample_token_index].reshape(-1, 1)
                # # print(sample_token)
                # token = torch.cat([token, sample_token], dim=1)
                
                if len(token[0]) == 75 or token[0, -1] == 50256:
                    break
            
            token = list(token[0][1:].detach().cpu().numpy())
            text = encoding.decode(token)
            text = re.split(r'[. ]', text)
            text = (list(filter(None, text)))
            # print(text)
            
            # paint the map

            img = np.array(transforms.Resize((378, 378))(Image.open(file[0])))
            fig, axes = plt.subplots(1, len(text)+1, figsize=(16, 4))
            axes[0].imshow(img)
            axes[0].set_xlabel(f'{text[-1]}')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            for i in range(1, len(text)+1):
                
                map = attention_map[i].reshape(1, 27, 27)
                map = nn.functional.interpolate(map.unsqueeze(0), scale_factor=14, mode='bilinear')[0].detach().cpu().numpy().reshape(378, 378, 1)
                axes[i].imshow(img)
                axes[i].imshow(map, alpha=0.6, cmap='rainbow')
                axes[i].set_xlabel(f'{text[i-1]}')
                axes[i].set_xticks([])
                axes[i].set_yticks([])
            
            # for i in range(len(text), 16):
            #     axes[i].remove()
            
            plt.savefig(f'{index}')
            # plt.show()