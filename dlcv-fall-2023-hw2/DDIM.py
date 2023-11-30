import torch
import torch.nn as nn
import torchvision
import numpy as np
import glob
import os
from PIL import Image
from utils import beta_scheduler
from UNet import UNet
import sys
import matplotlib.pyplot as plt

class DDIM(nn.Module):
    def __init__(self, device='cuda', seq=50, T=1000):
        self.device = device
        self.beta = beta_scheduler().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.seq = seq
        self.interval = int(T / seq)
        self.T = T
        # self.beta_tau = self.beta[np.arange(0, T, T / seq)].to(device)
        # self.alpha_tau = 1 - self.beta_tau
        # self.alpha_hat_tau = torch.cumprod(self.alpha_tau, dim=0)
        # print(len(self.beta_tau), len(self.alpha_tau), len(self.alpha_hat_tau))
    
    
    def sample_ddim(self, model, x):
        
        # x have to first load the noise.pt to tensor
        x = x.to(self.device)
        
        model.eval()
        with torch.no_grad():
            self.seq = 50
            # t = torch.linspace(0, self.T, (self.seq + 1)).to(self.device).long()    # inclusive the start, end and mid have (step - 2) 
                                                                                    # so total (self.seq + 1) number
            a = list()
            a.append(0)
            for i in range(1, 1000, 20):
                a.append(i)
            t = torch.tensor(a).to('cuda')
            # print(t)
            for i in range(self.seq+1, 1, -1):
                # print(i - 1)
                cur_t = t[i - 1]
                prev_t = t[i - 2]
                # print(cur_t, prev_t)
                
                sqrt_alpha_hat_prev = torch.sqrt(self.alpha_hat[prev_t]) if prev_t != 0 else 1
                sqrt_alpha_hat_cur = torch.sqrt(self.alpha_hat[cur_t])
                sqrt_1_minus_alpha_hat_cur = torch.sqrt(1 - self.alpha_hat[cur_t])
                sqrt_1_minus_alpha_hat_prev = torch.sqrt(1 - self.alpha_hat[prev_t]) if prev_t !=0 else 0
                
                pred_noise = model(x, cur_t.repeat(x.shape[0]))
                
                x = sqrt_alpha_hat_prev * (x - sqrt_1_minus_alpha_hat_cur * pred_noise) / sqrt_alpha_hat_cur \
                    + sqrt_1_minus_alpha_hat_prev * pred_noise
            
            
                
            return x
    
    
    # def sample_ddpm(self, model, x):
    
    #     model.eval()
    #     with torch.no_grad():
    #         img = None
    #         for i in range(self.T, 0, -1):
    #             print(i)
                
    #             t = torch.tensor([i]).to(self.device)
    #             t = t.repeat(1, 1, 1, 1)
                
    #             z = torch.randn(1, 3, 256, 256).to(self.device) if i > 1 else 0
                
    #             pred_noise = model(x, t)
                
    #             x = (1 / torch.sqrt(self.alpha[i-1])) * (x - pred_noise * (1 - self.alpha[i-1]) / torch.sqrt(1 - self.alpha_hat[i-1])) + torch.sqrt(self.beta[i-1]) * z
    #             # x = (1 / torch.sqrt(self.alpha[i-1])) * (x - pred_noise * (1 - self.alpha[i-1]) / torch.sqrt(1 - self.alpha_hat[i-1]))
    #             if i % 4 == 0:
    #                 # img.append(torchvision.utils.make_grid(x, nrow=10).detach().cpu().numpy())
    #                 if img is None:
    #                     img = x
    #                 else:
    #                     img = torch.cat([img, x], dim=0)
            
    #     return img
        
if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    noise_pth = sys.argv[1]
    output_pth = sys.argv[2]
    pretrained_model_weight = sys.argv[3]
    
    os.makedirs(output_pth, exist_ok=True)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_pth = sorted(glob.glob(os.path.join(noise_pth, "*.pt")))
    # # print(noise_pth)
    noise = torch.cat([torch.load(pth) for pth in noise_pth], dim=0).to(DEVICE)
    # print(noise.shape)   # 10, 3, 256, 256
    
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(pretrained_model_weight))
    ddim = DDIM(device=DEVICE)
    
    img = ddim.sample_ddim(model, noise)
    
    output_img_pth = [os.path.splitext(os.path.basename(pth))[0] for pth in noise_pth]
    for i in range(len(output_img_pth)):
        torchvision.utils.save_image(img[i], f'{output_pth}/{output_img_pth[i]}.png', normalize=True, range=(-1, 1))
        


    # loss_fn = torch.nn.MSELoss()
    # img = (img - img.min()) / (img.max() - img.min())
    # # img = (((img.clamp(-1, 1) + 1) / 2) * 255).cpu().numpy().astype(np.uint8)
    # # print(img.shape)
    # img = np.transpose(img, axes=(0, 2, 3, 1)).astype(np.uint8)
    # for i in range(10):
    #     pil_image = Image.fromarray(img[i])

    #     # Save the PIL Image to a PNG file
    #     pil_image.save(f'{output_pth}/0{i}.png')