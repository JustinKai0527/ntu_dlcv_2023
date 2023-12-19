import os, sys
from opt import get_opts
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import dataset_dict
import torchvision
# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays
from metrics import *
# optimizer, scheduler
from utils import *
import cv2
# losses
from losses import loss_dict

# metrics
from metrics import *

from dataset import KlevrDataset
from torch.utils.data import DataLoader
from collections import defaultdict

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

#NERF
class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)
        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        # if num gpu is 1, print model structure and number of params
        if self.hparams.num_gpus == 1:
            for i, model in enumerate(self.models):
                name = 'coarse' if i == 0 else 'fine'
                print('number of %s model parameters : %.2f M' % 
                      (name, sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
        
        # # load model if checkpoint path is provided
        # if self.hparams.ckpt_path != '':
        #     print('Load model from', self.hparams.ckpt_path)
        #     load_ckpt(self.model, self.hparams.ckpt_path, self.hparams.prefixes_to_ignore)

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk) # chunk size is effective in val mode

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k in results:
            results[k] = torch.cat(results[k], 0)
        return results


    def test_step(self, batch, batch_nb):
        
        # the batch for test only have rays
        sample = batch

        # results = self(sample['rays'].squeeze())
        # depth = visualize_depth(results[f'depth_fine'].view(256, 256)) # (3, H, W)
        # torchvision.utils.save_image(depth, f"depth_visualize/{sample['image_id'][0]}.png")
        rgbs = self(sample["rays"].squeeze())['rgb_fine']
        torchvision.utils.save_image(rgbs.transpose(0, 1).reshape(3, 256, 256), f"{output_folder}/{sample['image_id'][0]}.png")
        # torchvision.utils.save_image(sample['rgbs'].squeeze().transpose(0, 1).reshape(3, 256, 256), f"gt/{sample['image_id'][0]}.png")
        return


if __name__ == '__main__':

    metadata_folder = sys.argv[1]
    output_folder = sys.argv[2]
    os.makedirs(output_folder, exist_ok=True)
    hparams = get_opts()[0]
    
    system = NeRFSystem(hparams)
    system.load_state_dict(torch.load("final_model.ckpt")["state_dict"])
    
    logger = TensorBoardLogger(save_dir='logs',
              name=hparams.exp_name,
              )
    
    trainer = Trainer(max_epochs=4,
                    logger=logger,
                    num_sanity_val_steps=1,
                    # benchmark=True)
    )
    test_data = KlevrDataset(metadata_folder, split='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    # trainer.fit(system)
    trainer.test(system, test_loader)
    
    