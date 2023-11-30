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
            t = torch.linspace(0, self.T, (self.seq + 1)).to(self.device).long()    # inclusive the start, end and mid have (step - 2) 
                                                                                    # so total (self.seq + 1) number
            a = list()
            a.append(0)
            for i in range(1, 1000, 20):
                a.append(i)
            t = torch.tensor(a).to('cuda')
            print(t)
            for i in range(self.seq+1, 1, -1):
                print(i - 1)
                cur_t = t[i - 1]
                prev_t = t[i - 2]
                print(cur_t, prev_t)
                
                sqrt_alpha_hat_prev = torch.sqrt(self.alpha_hat[prev_t]) if prev_t != 0 else 1
                sqrt_alpha_hat_cur = torch.sqrt(self.alpha_hat[cur_t])
                sqrt_1_minus_alpha_hat_cur = torch.sqrt(1 - self.alpha_hat[cur_t])
                sqrt_1_minus_alpha_hat_prev = torch.sqrt(1 - self.alpha_hat[prev_t]) if prev_t !=0 else 0
                
                pred_noise = model(x, cur_t.repeat(x.shape[0]))
                
                x = sqrt_alpha_hat_prev * (x - sqrt_1_minus_alpha_hat_cur * pred_noise) / sqrt_alpha_hat_cur \
                    + sqrt_1_minus_alpha_hat_prev * pred_noise
            
            return x
        
if __name__ == "__main__":
    
    noise_pth = "hw2_data/face/noise"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # only want to create a interpolation between 00.pt, and 01.pt
    noise_pth = sorted(glob.glob(os.path.join(noise_pth, "*.pt")))[:2]
    
    noise = torch.cat([torch.load(pth) for pth in noise_pth], dim=0).to(DEVICE)

    theta = torch.acos(torch.sum(noise[0] * noise[1]) / (torch.norm(noise[0]) * torch.norm(noise[1])))

    interpolate_noise = None    
    alpha_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    # spherical linear interpolation
    for alpha in alpha_list:
        # print(alpha)
        noise_alpha = (torch.sin((1 - alpha) * theta) * noise[0] / torch.sin(theta) \
                        + torch.sin(alpha * theta) * noise[1] / torch.sin(theta)).reshape(-1, *noise.shape[1:])
        if interpolate_noise is None:
            interpolate_noise = noise_alpha
        else:
            interpolate_noise = torch.cat([interpolate_noise, noise_alpha], dim=0)    
            
            
    # linear interpolation
    # for alpha in alpha_list:

    #     noise_alpha = (1 - alpha) * noise[0] + alpha * noise[1]
    #     noise_alpha = noise_alpha.reshape(-1, *noise.shape[1:])
        
    #     if interpolate_noise is None:
    #         interpolate_noise = noise_alpha
    #     else:
    #         interpolate_noise = torch.cat([interpolate_noise, noise_alpha], dim=0)
        
    
    print(interpolate_noise.shape)   # 11, 3, 256, 256
    
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load("hw2_data/face/UNet.pt"))
    ddim = DDIM(device=DEVICE)
    
    
    img = ddim.sample_ddim(model, interpolate_noise)
    
    # loss_fn = torch.nn.MSELoss()
    print(img.shape)
    img = (img.clamp(-1, 1) + 1 ) / 2
    torchvision.utils.save_image(torchvision.utils.make_grid(img, nrow=11), "spherical_linear_interpolation.png")
        

    