import torch
from diffusion import DDPM
from model import Conditional_Denoised_Unet
import matplotlib.pyplot as plt
from mnistm_dataloader import inverse_transform
import torchvision
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ddpm = DDPM(Conditional_Denoised_Unet(), device=DEVICE)
ddpm.load_state_dict(torch.load("best_model.pt"))
ddpm.to(DEVICE)
size = [3, 28, 28]
N = 3
y = 0
ddpm.sample(N, size, DEVICE, y)

# plt.imshow(torchvision.utils.make_grid(inverse_transform(img)).permute(1, 2, 0).cpu().numpy() / 255)
# plt.show()
