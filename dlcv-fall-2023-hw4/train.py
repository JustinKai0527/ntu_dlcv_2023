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

# optimizer, scheduler
from utils import *

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

    def prepare_data(self):
        self.train_dataset = KlevrDataset(root_dir='dataset/', split='train', get_rgb=True)
        self.val_dataset = KlevrDataset(root_dir='dataset/', split='val', get_rgb=True)


    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=4, batch_size=self.hparams.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        log['train/loss'] = loss = self.loss(results, rgbs)
        
        with torch.no_grad():
            if 'rgb_fine' in results:
                psnr_ = psnr(results['rgb_fine'], rgbs)
            else:
                psnr_ = psnr(results['rgb_coarse'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            torchvision.utils.save_image(img, f"output/{batch['image_id'][0]}.png")
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)
        
        W, H = self.hparams.img_wh
        img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
        img = img.permute(2, 0, 1) # (3, H, W)
        torchvision.utils.save_image(img, f"output/{batch['image_id'][0]}.png")
        
        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        values = dict()
        for k, v in log.items():
            values[k] = v
        self.log_dict(values)
        
        return self._log_hyperparams
    
    # def validation_epoch_end(self, outputs):
    #     mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

    #     return {'progress_bar': {'val_loss': mean_loss,
    #                              'val_psnr': mean_psnr},
    #             'log': {'val/loss': mean_loss,
    #                     'val/psnr': mean_psnr}
    #            }


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)
    
    checkpoint_callback = ModelCheckpoint(dirpath = f'ckpt/{hparams.exp_name}',
                                          filename = 'best_model',
                                          monitor='val_psnr',
                                          mode='max',
                                          save_top_k=3,)
    
    logger = TensorBoardLogger("logs", name="my_exp_name")

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=checkpoint_callback,
                      logger=logger,
                      #early_stop_callback=None,
                      #weights_summary=None,
                      #progress_bar_refresh_rate=1,
                      #gpus=hparams.num_gpus,
                      #distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=0,
                      benchmark=True,
                      precision=16,
                      #profiler=True,
                      #amp_level='O1'
                      )

    trainer.fit(system)
    