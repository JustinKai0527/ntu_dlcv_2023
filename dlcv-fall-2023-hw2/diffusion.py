import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np
from mnistm_dataloader import Mnistm_dataset, inverse_transform
import pandas as pd
import matplotlib.pyplot as plt
from DDPM_Unet import Conditional_Denoised_Unet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class DDPM(nn.Module):
    def __init__(self, model, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, img_size=28, device='cuda', drop_prob=0.1):
        super(DDPM, self).__init__()
        
        self.noise_steps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.drop_prob = drop_prob
        self.model = model
        
        self.alpha_hat = self.prepare_cosine_noise_schedule().to(device)                                # 1002 shape
        self.beta = torch.clamp((1 - self.alpha_hat[1:] / self.alpha_hat[:-1]), max=0.999)       # 1001 shape
        self.alpha = 1 - self.beta
        self.loss = nn.MSELoss()
        # self.beta = self.prepare_linear_noise_schedule()
        # self.alpha = 1 - self.beta
        # self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        # plt.plot(range(len(self.alpha_hat)), self.alpha_hat)
        # plt.plot(range(1001), self.prepare_cosine_noise_schedule())
        # plt.show()
    # # linear noise schedule 
    def prepare_linear_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps + 1)    # 1001 shape
    
    def prepare_cosine_noise_schedule(self):
        t = torch.linspace(0, self.noise_steps, self.noise_steps+2)     # 1002 shape
        s = 0.002
        f_t = torch.square(torch.cos(torch.pi * (t / self.noise_steps + s) / (1 + s) / 2))
        alpha_t = f_t / f_t[0]
        
        return alpha_t
    
    def get_noise_img(self, img, t):
        
        # t is (B, )   each img t is not same
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).reshape(-1, 1, 1, 1)
        sqrt_1_minus_alpha_hat = torch.sqrt((1 - self.alpha_hat[t])).reshape(-1, 1, 1, 1)
        
        noise = torch.randn_like(img).to(self.device)
        
        return ((sqrt_alpha_hat * img) + sqrt_1_minus_alpha_hat * noise), noise    # return the current noise to predict it
    
    def get_uniform_time_step(self, B):
        return torch.randint(low=1, high=self.noise_steps, size=(B,)).to(self.device)
    
    def forward(self, x, y):
        
        t = self.get_uniform_time_step(x.shape[0])

        x_t, noise = self.get_noise_img(x, t)

        mask = torch.bernoulli(torch.ones_like(y) * self.drop_prob).to(self.device)
        # print(x_t.dtype, y.dtype, t.dtype, mask.dtype)              float, long, float, float
        return self.loss(noise, self.model(x_t, y, t / self.noise_steps, mask))
        
    def sample(self, N, size, device, y, guide_w = 0.0):    # y is (N, )
        
        x = torch.randn(N, *size).to(device)
        y = torch.tensor([0]).to(device).repeat(N)
        
        mask = torch.zeros_like(y).to(device)

        for i in range(self.noise_steps, 0, -1):
            print(i)
            t = torch.tensor([i / self.noise_steps]).repeat(N).to(device).reshape(-1, 1, 1, 1)
            z = torch.randn(N, *size).to(device) if i > 1 else 0

            x = (1 / self.alpha[i]) * (x - (((1 - self.alpha[i]) / torch.sqrt(1 - self.alpha_hat[i])) * self.model(x, y, t, mask))) + torch.sqrt(self.beta[i]) * z
        
        return x


