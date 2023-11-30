import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tokenizer import BPETokenizer
import timm
from attn_decoder import Decoder, Config
from PEFT_dataset import PEFT_dataset
from tqdm import tqdm
import glob
from PIL import Image
import os
import json

# dataset for inference
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
# from torch.utils.tensorboard import SummaryWriter

# hyper-parameter
train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
valid_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
NUM_EPOCH = 8
best_loss = 100
lr = 1e-4
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

cfg = Config("hw3_data/p2_data/decoder_model.bin")
encoder = timm.create_model('vit_large_patch14_clip_224', pretrained=True)
decoder = Decoder(cfg)
train_dataset = PEFT_dataset("hw3_data/p2_data/images/train", train_tfm, json_file_pth="hw3_data/p2_data/train.json")
valid_dataset = test_dataset("hw3_data/p2_data/images/val", valid_tfm)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

# fixing the encoder
for param in encoder.parameters():
    param.requires_grad = False
    
state_dict = torch.load("hw3_data/p2_data/decoder_model.bin")
fix_param = [key for key in state_dict]

for name, param in decoder.named_parameters():
    if any(name.endswith(key) for key in fix_param):
#         # print(name)
        param.requires_grad = False
#         pass
#     else:
#         print(name)
# update_param = {name: param for name, param in decoder.state_dict().items() if any(name.endswith(key) for key in fix_param)}
optimizer = torch.optim.Adam(decoder.parameters(), lr)
# for key in update_param:
#     print(key)
encoder.to(DEVICE)
decoder.to(DEVICE)
print("Total params:", sum(p.numel() for p in decoder.parameters() if p.requires_grad))
encoding = BPETokenizer("encoder.json", "vocab.bpe")
# writer = SummaryWriter()
for ep in range(1, NUM_EPOCH+1):
    print(f"Epoch: {ep}")

    decoder.train()    
    total_loss = []
    for img, token, gt in tqdm(train_loader):
        
        img = img.to(DEVICE)
        token = token.to(DEVICE)
        gt = gt.to(DEVICE).long()
        optimizer.zero_grad()
        enc_feat = encoder.forward_features(img).narrow(1, 1, 256)
        # print(enc_feat.shape)
        pred = decoder(token, enc_feat)

        loss = loss_fn(pred.reshape(-1, 50257), gt.reshape(-1))
        # print(loss.item())
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        # print(encoding.decode(list(pred.argmax(dim=2)[0].detach().cpu().numpy())))
        
        
    total_loss = sum(total_loss) / len(total_loss)
    print(f"Total Loss: {total_loss}")
    # writer.add_scalar("Loss/Train", total_loss, ep)

    
    update_param = {name: param for name, param in decoder.state_dict().items() if any(name.endswith(key) for key in fix_param)}
    torch.save(update_param, f"model/model_{ep}.pt")
    torch.save(decoder.state_dict(), f"decoder/model_{ep}.pt")
    
    encoder.eval()
    decoder.eval()
    # with torch.no_grad():
        
    #     total_loss = []
    #     for img, token, gt in tqdm(valid_loader):
    #         img = img.to(DEVICE)
    #         token = token.to(DEVICE)
    #         gt = gt.to(DEVICE).long()
            
    #         enc_feat = encoder.forward_features(img).narrow(1, 1, 576)
    #         pred = decoder(token, enc_feat)
    #         loss = loss_fn(pred.reshape(-1, 50257), gt.reshape(-1))

    #         total_loss.append(loss.item())
    
    #     total_loss = sum(total_loss) / len(total_loss)
    #     if total_loss <= best_loss:
    #         best_loss = total_loss
    #         print(total_loss)
    #         print("best model")
    
    # with torch.no_grad():

    #     result_dict = {}
    #     for img, file in tqdm(valid_loader):
            
    #         img = img.to(DEVICE)
    #         enc_feat = encoder.forward_features(img).narrow(1, 1, 729)
    #         token = torch.tensor([50256]).reshape(-1, 1).to(DEVICE)
            
    #         while True:
                
    #             pred_token = decoder(token, enc_feat)
    #             pred_token = nn.Softmax(dim=2)(pred_token)
    #             # print(torch.topk(pred_token, 3, dim=2))
    #             pred_token = pred_token.argmax(dim=2)
    #             token = torch.cat([token, pred_token[0, -1].reshape(-1, 1)], dim=1)
                
    #             if len(token[0]) == 75 or token[0, -1] == 50256:
    #                 break
                
    #         # print(token[0, 1:])
    #         token = list(token[0, 1:-1].detach().cpu().numpy())
    #         text = encoding.decode(token)
    #         result_dict[file[0]] = text
            
    #     output_json_pth = f"output_json_file/huge{ep}.json"
    #     with open(output_json_pth, 'w') as json_file:
    #         json.dump(result_dict, json_file, indent=4)
    #     writer.add_scalar("Loss/Valid", total_loss, ep)
        
    #     if total_loss < best_loss:
    #         best_loss = total_loss
    #         print("Save Model")
    #         update_param = {name: param for name, param in decoder.state_dict().items() if any(name.endswith(key) for key in fix_param)}
    #         torch.save(update_param, "adapter.pt")
            
# cfg = Config("hw3_data/p2_data/decoder_model.bin")
# decoder = Decoder(cfg)
# # models = timm.list_models('vit_base_patch32_clip_256*')
# # print(models)
# # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L661
# model = timm.create_model('vit_base_patch32_clip_256', pretrained=True)
# x = torch.randn((10, 3, 256, 256))
# x = model.forward_features(x)
# # print(x.shape)   # got B, S, E     batch_size, patch_size, embed_dim
# token = torch.randint(low=0, high=50, size=(10, 10))
# out = decoder(token, x)
# print(out.shape)        # torch.Size([10, 10, 50257])

