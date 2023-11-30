import torch
import torchvision
from DDPM_Unet import Conditional_Denoised_Unet
from DDPM_train import DDPM
import numpy as np
import imageio

if __name__ == "__main__":
    
    torch.manual_seed(50)
    time_step = 400
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ddpm = DDPM(Conditional_Denoised_Unet(), time_step, DEVICE, drop_prob=0)
    ddpm.load_state_dict(torch.load("ddpm_model/model_75.pt"))
    ddpm.to(DEVICE)
    
    N = 100
    n_cls = torch.arange(0, 10).reshape(-1, 1).repeat(1, 10).reshape(-1).to('cuda')
    print(n_cls)

    with torch.no_grad():
        
        img, first_img = ddpm.sample(N, device='cuda', cls_free_guide_w=1.5, y=n_cls)
        torchvision.utils.save_image(torchvision.utils.make_grid(img, nrow=10), "100img.png")
        first_img = torch.from_numpy(np.array(first_img))
        torchvision.utils.save_image(torchvision.utils.make_grid(first_img, nrow=6), "0img.png")
        