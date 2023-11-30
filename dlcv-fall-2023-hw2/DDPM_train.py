import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from mnistm_dataloader import Mnistm_dataset, inverse_transform
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from tqdm import tqdm
from DDPM_Unet import Conditional_Denoised_Unet
# from torch.utils.tensorboard import SummaryWriter

def get_linear_noise_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    return betas

def get_cosine_noise_schedule(num_time_step, device):
    
    x = torch.linspace(0, num_time_step, num_time_step+2).to(device)
    s = 0.008
    alpha_hat = torch.cos(((x / num_time_step) + s) * torch.pi / 2).to(device) ** 2
    alpha_hat = alpha_hat / alpha_hat[0]
    beta = 1 - (alpha_hat[1:] / alpha_hat[:-1])
    beta = torch.clip(beta, min=0.0001, max=0.9999)
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    sqrt_alpha_hat = torch.sqrt(alpha_hat)
    sqrt_1_minus_alpha_hat = torch.sqrt(1 - alpha_hat)
    sqrt_alpha = torch.sqrt(alpha)
    sqrt_1_minus_alpha = (torch.sqrt(1 - alpha))
    sqrt_beta = torch.sqrt(beta)
    
    return {"beta": beta,
            "alpha": alpha,
            "alpha_hat": alpha_hat,
            "sqrt_alpha": sqrt_alpha,
            "sqrt_alpha_hat": sqrt_alpha_hat,
            "sqrt_beta": sqrt_beta,
            "sqrt_1_minus_alpha": sqrt_1_minus_alpha,
            "sqrt_1_minus_alpha_hat": sqrt_1_minus_alpha_hat}


class DDPM(nn.Module):
    def __init__(self, model, num_time_step, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        
        self.model = model.to(device)
        self.num_time_step = num_time_step
        self.device = device
        self.drop_prob=drop_prob

        for k, v in get_cosine_noise_schedule(self.num_time_step, self.device).items():
            self.register_buffer(k, v)
        self.loss_fn = nn.MSELoss()
    
    def get_uniform_time_step(self, N):
        
        return torch.randint(1, self.num_time_step+1, size=(N, )).to(self.device)
    
    def get_noise_img(self, x, t):
        
        noise = torch.randn_like(x).to(self.device)
        
        return (self.sqrt_alpha_hat[t, None, None, None] * x + self.sqrt_1_minus_alpha_hat[t, None, None, None] * noise), noise
    
    def forward(self, x, y):
        
        t = self.get_uniform_time_step(x.shape[0])

        x_t, noise = self.get_noise_img(x, t)
        
        # torchvision.utils.save_image(torchvision.utils.make_grid(x_t), "test.png")
        mask = torch.bernoulli(torch.ones_like(y) * self.drop_prob).to(self.device)
        
        return self.loss_fn(noise, self.model(x_t, y, (t / self.num_time_step), mask))
        
    def sample(self, N, device, img_dim=(3, 28, 28), cls_free_guide_w=0.0, y=None):
        
        x = torch.randn(N, *img_dim).to(device)
        if y is None:
            y = torch.arange(0, 10).to(device).repeat(int(N / 10))
        else:       
            y = torch.tensor([y]).repeat(N).to(device)
            
        y = y.repeat(2)
        mask = torch.zeros_like(y).to(device)
        mask[N:] = 1
        
        img = []
        for i in range(self.num_time_step, 0, -1):
            # print(i)
            
            t = torch.tensor([i / self.num_time_step]).to(device)
            t = t.repeat(N, 1, 1, 1)
            
            z = torch.randn(N, *img_dim).to(device) if i > 1 else 0
            
            x = x.repeat(2, 1, 1, 1)
            t = t.repeat(2, 1, 1, 1)
            pred_noise = self.model(x, y, t, mask)
            
            cls_pred_noise = pred_noise[:N]
            cls_free_pred_noise = pred_noise[N:]
            pred_noise = (1 + cls_free_guide_w) * cls_pred_noise - cls_free_guide_w * cls_free_pred_noise
            x = x[:N]
            
            x = (1 / self.sqrt_alpha[i]) * (x - pred_noise * (1 - self.alpha[i]) / self.sqrt_1_minus_alpha_hat[i]) + self.sqrt_beta[i] * z
            # if i % 4 == 0:
            #     img.append(torchvision.utils.make_grid(x, nrow=10).detach().cpu().numpy())
        #     if i % 80 == 0:
        #         print()
        #         print(i)
        #         img.append(x[0].detach().cpu().numpy())
        # img.append(x[0].detach().cpu().numpy())
        return x
    
if __name__ == "__main__":
    
    NUM_EPOCH = 75
    BATCH_SIZE = 256
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    time_step = 400
    
    n_cls = 10
    feature = 256
    lr = 1e-4
    cls_free_guide_w = [0.0, 0.5, 2.0]
    
    ddpm = DDPM(Conditional_Denoised_Unet(), time_step, DEVICE, drop_prob=0.1)
    
    tfm = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_pd = pd.read_csv("hw2_data/digits/mnistm/train.csv")
    train_filename = train_pd["image_name"].to_numpy()
    train_label = train_pd["label"].to_numpy()
    
    valid_pd = pd.read_csv("hw2_data/digits/mnistm/val.csv")
    valid_filename = valid_pd["image_name"].to_numpy()
    valid_label = valid_pd["label"].to_numpy()
    
    train_data = Mnistm_dataset("hw2_data/digits/mnistm/data", train_filename, train_label, tfm)
    valid_data = Mnistm_dataset("hw2_data/digits/mnistm/data", valid_filename, valid_label, tfm)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False)
    
    optimizer = torch.optim.Adam(ddpm.parameters(), lr)
    # writer = SummaryWriter()
    
    for ep in range(1, NUM_EPOCH+1):
        print(f"Epoch: {ep}")
        
        ddpm.train()        

        optimizer.param_groups[0]['lr'] = lr * (1 - ep / NUM_EPOCH)
        total_loss = []
        for x, y in tqdm(train_loader):
            
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            
            loss = ddpm(x, y)
            loss.backward()

            optimizer.step()
            total_loss.append(loss.item())
            
        print(sum(total_loss) / len(total_loss))
        # writer.add_scalar("Loss/Train", sum(total_loss) / len(total_loss), ep)
        
        ddpm.eval()
        
        with torch.no_grad():
            
            N = 4 * n_cls
            for i, w in enumerate(cls_free_guide_w):
                
                x_gen = ddpm.sample(N, DEVICE, cls_free_guide_w=w)
                x = torchvision.utils.make_grid(x_gen, nrow=10)
                torchvision.utils.save_image(x, f"gen_img/{ep}_{w}test.png")
        
            if ep % 1 == 0:
                print("Save Model")
                torch.save(ddpm.state_dict(), f"ddpm_model/model_{ep}.pt")
          
                
        # if ep % 1 == 0:
        #     print("Save Model")
        #     torch.save(ddpm.state_dict(), f"ddpm_model/model_{ep}.pt")
            
                