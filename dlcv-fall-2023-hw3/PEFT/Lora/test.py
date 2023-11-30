import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tokenizer import BPETokenizer
import timm
from lora_decoder import Decoder, Config
from PEFT_dataset import PEFT_dataset
from tqdm import tqdm
import glob
from PIL import Image
import os
import json
import loralib as lora
## dataset for inference
# class test_dataset(Dataset):
#     def __init__(self, root, tfm=None):
        
#         self.root = root
#         self.img_file = glob.glob(os.path.join(root, "*"))
#         self.tfm = tfm
#         self.len = len(self.img_file)
        
#     def __getitem__(self, index):
        
#         img = Image.open(self.img_file[index]).convert('RGB')
        
#         if self.tfm:
#             img = self.tfm(img)

#         return img, os.path.basename(self.img_file[index]).split('.')[0]
    
#     def __len__(self):
#         return self.len
# # from torch.utils.tensorboard import SummaryWriter

# # hyper-parameter
# train_tfm = transforms.Compose([
#     transforms.Resize((378, 378)),
#     transforms.ToTensor(),
# ])
# valid_tfm = transforms.Compose([
#     transforms.Resize((378, 378)),
#     transforms.ToTensor(),
# ])

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# BATCH_SIZE = 4
# NUM_EPOCH = 80
# best_loss = 100
# lr = 1e-4
# loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# cfg = Config("hw3_data/p2_data/decoder_model.bin")
# encoder = timm.create_model('vit_huge_patch14_clip_378', pretrained=True)
# decoder = Decoder(cfg)
# train_dataset = PEFT_dataset("hw3_data/p2_data/images/train", train_tfm, json_file_pth="hw3_data/p2_data/train.json")
# valid_dataset = test_dataset("hw3_data/p2_data/images/val", valid_tfm)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

# # fixing the encoder
# for param in encoder.parameters():
#     param.requires_grad = False
    
# state_dict = torch.load("hw3_data/p2_data/decoder_model.bin")
# fix_param = [key for key in state_dict]

# lora.mark_only_lora_as_trainable(decoder)    # first only let the lora requires_grad = True 
# for name, param in decoder.named_parameters():      # and add the cross-attn requires_grad = True
#     if not any(name.endswith(key) for key in fix_param):
# #         # print(name)
#         param.requires_grad = True
        
# optimizer = torch.optim.Adam(decoder.parameters(), lr)

# encoder.to(DEVICE)
# decoder.to(DEVICE)

# print("Total params:", sum(p.numel() for p in decoder.parameters() if p.requires_grad))

# torch.save(lora.lora_state_dict(decoder), "lora_model/model.pt")
# state_dict = torch.load("hw3_data/p2_data/decoder_model.bin")
# fix_param = [key for key in state_dict]

# for name, param in decoder.named_parameters():
#     if any(name.endswith(key) for key in fix_param):
# #         # print(name)
#         param.requires_grad = False

# param_store = {}
# for name, param in decoder.named_parameters():
#     if any(name.endswith(key) for key in fix_param):
#         pass
#     else:
#         param_store[name] = param

# torch.save(param_store, "test.pt")

state_dict = torch.load('model/model_4.pt')
print(sum([p.numel() for n, p in state_dict.items()]))
