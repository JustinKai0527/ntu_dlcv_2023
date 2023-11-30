import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tokenizer import BPETokenizer
import timm
from decoder import Decoder, Config
from PIL import Image
import json
import sys
import os
import glob
from tqdm import tqdm
import numpy as np

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

        return img, os.path.basename(self.img_file[index]).split('.')[0]
    
    def __len__(self):
        return self.len
# bash hw3_2.sh $1 $2 $3
# $1: path to the folder containing test images (e.g. hw3/p2_data/images/test/)
# $2: path to the output json file (e.g. hw3/output_p2/pred.json) 
# $3: path to the decoder weights (e.g. hw3/p2_data/decoder_model.bin)
# (This means that you don’t need to upload decoder_model.bin)

# def expand_beam(decoder, tokens, probs, num_beams, enc_feat):  
#     next_token = []
#     next_probs = []
#     for i, (token, score) in enumerate(zip(tokens, probs)):
#         # If the sequence already ends, keep it in the next_token
#         print(token)
#         if torch.any(token[-1] == 50256) and len(token) != 1:
#             next_token.append(token)
#             next_probs.append(score)
#             continue
        
#         # pred_token N, E
#         pred_token = decoder(token.reshape(1, -1), enc_feat)[0, -1, :].squeeze()

#         top_k_probs, top_k_indices = torch.topk(pred_token, num_beams, dim=0)
#         # print(top_k_probs.shape)     (num_beams,
#         for prob, token_index in zip(top_k_probs, top_k_indices):
#             # print(token, token_index)
#             new_token = torch.cat([token, torch.tensor([token_index]).to('cuda')])
#             new_score = score - torch.log(prob).item()
#             next_token.append(new_token)
#             next_probs.append(new_score)
            
#     return next_token, next_probs


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
    decoder.load_state_dict(torch.load("PEFT_final_model.pt"), strict=False)
    encoding = BPETokenizer("encoder.json", "vocab.bpe")
    state_dict = torch.load('PEFT_final_model.pt')
    print(sum([p.numel() for n, p in state_dict.items()]))
    # state_dict = torch.load(decoder_weight_pth)
    # fix_param = [key for key in state_dict]
    
    # for key in decoder.named_parameters():
    #     if not any(key.endswith(fix_key) for fix_key in fix_param):
    #         print(key)
    # save_param = {key: param for key, param in decoder.named_parameters() if not any(key.endswith(fix_key) for fix_key in fix_param)}
    # torch.save(save_param, "PEFT_adapter_final_model.pt")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.to(DEVICE)
    decoder.to(DEVICE)
    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        result_dict = {}
        for img, file in tqdm(dataloader):
            
            img = img.to(DEVICE)
            enc_feat = encoder.forward_features(img).narrow(1, 1, 729)
            token = torch.tensor([50256]).reshape(-1, 1).to(DEVICE)
            
            while True:
                
                pred_token = decoder(token, enc_feat)
                pred_token = nn.Softmax(dim=2)(pred_token)                 # make sure the prob is non-negative
                # gready search
                # print(torch.topk(pred_token, 3, dim=2))
                pred_token = pred_token.argmax(dim=2)
                token = torch.cat([token, pred_token[0, -1].reshape(-1, 1)], dim=1)
                
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
            
            # # beam search
            # num_beams = 5
            # token = torch.tensor([50256] * num_beams).reshape(-1, 1).to(DEVICE)
            # probs = [0] * num_beams
            # while True:
            #     # print("good start")
            #     token, probs = expand_beam(decoder, token, probs, num_beams, enc_feat)
                
            #     # print(token, probs)
            #     probs = np.array(probs)
            #     index = probs.argsort(axis=-1)[-num_beams:]

            #     token = [token[i] for i in index]
            #     probs = probs[index]
                
            #     if len(token[0]) == 75 or token[0][-1] == 50256:
            #         break
                
            # print(token[0, 1:])
            token = list(token[0][1:-1].detach().cpu().numpy())
            # print(token)
            text = encoding.decode(token)
            result_dict[file[0]] = text

        with open(output_json_file, 'w') as json_file:
            json.dump(result_dict, json_file, indent=4)
