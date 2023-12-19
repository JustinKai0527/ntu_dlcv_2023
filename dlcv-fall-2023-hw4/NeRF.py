from dataset import KlevrDataset
# from run_nerf_helpers import *
# from run_nerf import *
import torch
from torch.utils.data import DataLoader

print(torch.cuda.is_available())
dataset = KlevrDataset("dataset/dataset", split='val')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# print(dataset.img_wh)
# print(dataset.focal)
# print(dataset.scene_boundaries)
# print(dataset.directions.shape)
# print(dataset.near.shape, dataset.far.shape)


# train  rays: {rays_o, rays_d, near, far}  rgbs:  {r, g, b} 
# {'rays': tensor([-1.7035e+00, -2.7072e+00,  3.1523e+00,  1.5670e-01,  9.5058e-01, -2.6805e-01,  1.0000e-10,  3.8999e+00]), 'rgbs': tensor([0.8078, 0.8078, 0.8078])}
# val    rays   c2w  rgbs   valid_mask
# test   rays   c2w
# print(dataset.K)

for sample in dataloader:
    # print(sample['rays'][0, :])
    # print(torch.stack([sample['rays'][:, :3], sample['rays'][:, 3:6]], dim=0).shape)
    # print(sample['rays'].shape)
    print(sample['rgbs'].shape)
    break


# notice the shape of val & train is different