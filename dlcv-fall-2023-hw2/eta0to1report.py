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
    
    
    def sample_ddim(self, model, x, eta_list=None):
        
        # x have to first load the noise.pt to tensor
        x = x.to(self.device)
        
        model.eval()
        with torch.no_grad():
            self.seq = 50
            # t = torch.linspace(0, self.T, (self.seq + 1)).to(self.device).long()    # inclusive the start, end and mid have (step - 2) 
                                                                             # so total (self.seq + 1) number
            img = None
            tmp = x.clone().to(self.device)
            a = list()
            a.append(0)
            for i in range(1, 1000, 20):
                a.append(i)
            t = torch.tensor(a).to('cuda')
            
            for eta in eta_list:
                x = tmp
                for i in range(self.seq + 1, 1, -1):
                    print(i)
                    cur_t = t[i - 1]
                    prev_t = t[i - 2]
                    print(cur_t, prev_t)
                    prev_t = prev_t if prev_t > 0 else torch.tensor([1])
                    sigma = eta * (torch.sqrt((1 - self.alpha_hat[prev_t]) / (1 - self.alpha_hat[cur_t]))) * (torch.sqrt(1 - (self.alpha_hat[cur_t] / self.alpha_hat[prev_t])))
                    
                    sqrt_alpha_hat_prev = torch.sqrt(self.alpha_hat[prev_t])
                    sqrt_alpha_hat_cur = torch.sqrt(self.alpha_hat[cur_t])
                    sqrt_1_minus_alpha_hat_cur = torch.sqrt(1 - self.alpha_hat[cur_t])
                    sqrt_1_minus_alpha_hat_prev = torch.sqrt(1 - self.alpha_hat[prev_t] - sigma ** 2)
                    
                    pred_noise = model(x, cur_t.repeat(x.shape[0]))
                    z = torch.randn(*x.shape).to(self.device)
                    x = (sqrt_alpha_hat_prev * (x - sqrt_1_minus_alpha_hat_cur * pred_noise) / sqrt_alpha_hat_cur \
                        + sqrt_1_minus_alpha_hat_prev * pred_noise + sigma * z).float()
                    
                if img is None:
                    img = x
                else:
                    img = torch.cat([img, x], dim=0)
                    
            return img
    
    
    
if __name__ == "__main__":
    
    noise_pth = "hw2_data/face/noise"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_pth = sorted(glob.glob(os.path.join(noise_pth, "*.pt")))[:4]
    # # print(noise_pth)
    noise = torch.cat([torch.load(pth) for pth in noise_pth], dim=0).to(DEVICE)
    print(noise.shape)   # 4, 3, 256, 256
    
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load("hw2_data/face/UNet.pt"))
    ddim = DDIM(device=DEVICE)
    
    ETA = [0, 0.25, 0.5, 0.75, 1]
    img = ddim.sample_ddim(model, noise, eta_list=ETA)
    
    # loss_fn = torch.nn.MSELoss()
    print(img.shape)
    img = (img.clamp(-1, 1) + 1 ) / 2
    torchvision.utils.save_image(torchvision.utils.make_grid(img, nrow=4), "report.png")
        

    