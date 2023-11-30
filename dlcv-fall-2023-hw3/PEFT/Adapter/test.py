import json
import timm
import torch

print(timm.list_models("vit*large*"))
# # Your JSON file path
# # json_file_path = 'test.json'

# # # Open and read the JSON file
# # with open(json_file_path, 'r') as json_file:
# #     # Load the JSON data into a dictionary
# #     data_dict = json.load(json_file)

# # # Now data_dict is a Python dictionary
# # print(data_dict)

# a = torch.tensor(torch.randn((3,3)) * 4)
# for i in a:
#     print(i)

state_dict = torch.load('PEFT_final_model.pt')
print(sum([p.numel() for n, p in state_dict.items()]))


# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# from tokenizer import BPETokenizer
# import timm
# from decoder import Decoder, Config
# from PEFT_dataset import PEFT_dataset
# from tqdm import tqdm
# import glob
# from PIL import Image
# import os
# import json



# cfg = Config("hw3_data/p2_data/decoder_model.bin")
# encoder = timm.create_model('vit_huge_patch14_clip_378', pretrained=True)
# decoder = Decoder(cfg)
# decoder.load_state_dict(torch.load("PEFT_final_model.pt"), strict=False)
# # fixing the encoder
# for param in encoder.parameters():
#     param.requires_grad = False
    
# state_dict = torch.load("hw3_data/p2_data/decoder_model.bin")
# fix_param = [key for key in state_dict]

# for name, param in decoder.named_parameters():
#     if any(name.endswith(key) for key in fix_param):
# #         # print(name)
#         param.requires_grad = False
# #         pass
# #     else:
# #         print(name)

# # for key in update_param:
# #     print(key)
# print("Total params:", sum(p.numel() for p in decoder.parameters() if p.requires_grad))

# # for name, param in decoder.named_parameters():
# #     if any(name.endswith(key) for key in fix_param):
# #         print(name)
        
# param_store = {}
# for name, param in decoder.named_parameters():
#     if any(name.endswith(key) for key in fix_param):
#         print(name)
#     else:
#         param_store[name] = param

# torch.save(param_store, "test.pt")