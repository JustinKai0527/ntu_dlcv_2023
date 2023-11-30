import torch
import torchvision
from DDPM_Unet import Conditional_Denoised_Unet
from DDPM_train import DDPM
import os
import sys
import numpy as np
import imageio

def save_img(x, output_pth, cls):
    
    for i in range(1, 101):
        img = x[i-1].squeeze()
        number = str(i).zfill(3)
        torchvision.utils.save_image(img, f"{output_pth}/{cls}_{number}.png")
    
if __name__ == "__main__":
    
    seed = 50
    torch.manual_seed(seed)
    np.random.seed(seed)
    time_step = 400
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ddpm = DDPM(Conditional_Denoised_Unet(), time_step, DEVICE, drop_prob=0)
    ddpm.load_state_dict(torch.load("hw2_model/DDPM_model.pt"))
    ddpm.to(DEVICE)
    
    output_pth = sys.argv[1]
    N = 100
    n_cls = torch.arange(0, 10)
    os.makedirs(output_pth, exist_ok=True)

    with torch.no_grad():
        
        for cls in n_cls:
            print(f"sample {cls}")
            x_gen = ddpm.sample(100, DEVICE, cls_free_guide_w=1.5, y=cls)
            save_img(x_gen, output_pth, cls)
            
    
    