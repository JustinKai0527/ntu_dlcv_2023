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
            
            t = torch.linspace(0, self.T, (self.seq + 1)).to(self.device).long()    # inclusive the start, end and mid have (step - 2) 
                                                                             # so total (self.seq + 1) number
            
            for i in range(self.seq + 1, 1, -1):
                print(i)
                cur_t = t[i - 1] - 1
                prev_t = t[i - 2] - 1
                
                prev_t = prev_t if prev_t > 0 else torch.tensor([1])
                
                sqrt_alpha_hat_prev = torch.sqrt(self.alpha_hat[prev_t])
                sqrt_alpha_hat_cur = torch.sqrt(self.alpha_hat[cur_t])
                sqrt_1_minus_alpha_hat_cur = torch.sqrt(1 - self.alpha_hat[cur_t])
                sqrt_1_minus_alpha_hat_prev = torch.sqrt(1 - self.alpha_hat[prev_t])
                
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
    

    gt_pth = sorted(glob.glob(os.path.join("hw2_data/face/GT", "*.png")))
    gt = np.array([np.array(Image.open(pth)) for pth in gt_pth])
    pred_pth = sorted(glob.glob(os.path.join("hw2_DDIM/test", "*.png")))
    pred = np.array([np.array(Image.open(pth)) for pth in pred_pth])
    print(pred.shape, gt.shape)
    for i in range(10):
        print(np.mean(np.square(pred[i] - gt[i])))
        
    print(np.mean(np.square(pred - gt)))