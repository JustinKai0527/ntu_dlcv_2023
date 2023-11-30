import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, chin, chout):
        super(DoubleConv, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(chin, chout, 3, 1, 1),
            nn.GroupNorm(1, chout),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(chout, chout, 3, 1, 1),
            nn.GroupNorm(1, chout),
            nn.GELU(),
        )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        if x.shape[1] == x2.shape[1]:
            x2 = x2 + x
            return x2 / 1.414
        elif x1.shape[1] == x2.shape[1]:
            x2 = x2 + x1
            return x2 / 1.414
        
        return x2
    

class Down(nn.Module):
    def __init__(self, chin, chout):
        super(Down, self).__init__()
        
        self.block = nn.Sequential(
            DoubleConv(chin, chout),
            nn.MaxPool2d(2),
        )
    
    def forward(self, x):
        return self.block(x)
    
        
class Up(nn.Module):
    def __init__(self, chin, chout):
        super(Up, self).__init__()
        
        self.block = nn.Sequential(
            nn.ConvTranspose2d(chin, chout, 2, 2),       # H, W -> 2H, 2W
            DoubleConv(chout, chout),
            DoubleConv(chout, chout),
        )
        
    def forward(self, x, skip_x):

        x = torch.cat((x, skip_x), dim=1)
        x = self.block(x)
        return x

class Embedding(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(Embedding, self).__init__()
        
        self.input_dim = input_dim
        self.emb_layer = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        
    def forward(self, x):   # t (N, ) reshape to (N, 1)  y reshape to (N, n_cls)
        
        x = x.reshape(-1, self.input_dim)
        return self.emb_layer(x)
    
    
class Conditional_Denoised_Unet(nn.Module):
    def __init__(self, chin=3, feature=256, n_cls=10):
        super(Conditional_Denoised_Unet, self).__init__()
        
        self.chin = chin
        self.feature = feature
        self.n_cls = n_cls
        
        self.first_conv = DoubleConv(chin, feature)
        
        self.down = nn.ModuleList()
        self.down.append(Down(feature, feature))
        self.down.append(Down(feature, feature*2))
        
        self.to_latent = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        
        self.tim_emb1 = Embedding(1, 2 * feature)
        self.tim_emb2 = Embedding(1, feature)
        self.label_emb1 = Embedding(n_cls, 2 * feature)
        self.label_emb2 = Embedding(n_cls, feature)
        
        self.up = nn.ModuleList()
        self.up.append(nn.Sequential(
            nn.ConvTranspose2d(2 * feature, 2 * feature, 7, 7),
            nn.GroupNorm(8, 2 * feature),
            nn.ReLU(),
        ))
        
        self.up.append(Up(4 * feature, feature))
        self.up.append(Up(2 * feature, feature))
        
        self.out = nn.Sequential(
            nn.Conv2d(2 * feature, feature, 3, 1, 1),
            nn.GroupNorm(8, feature),
            nn.ReLU(),
            nn.Conv2d(feature, chin, 3, 1, 1)
        )
        
    def forward(self, x, y, t, mask):
        # y torch.long t torch.flaot mask torch.float
        
        x = self.first_conv(x)

        x1 = self.down[0](x)
        x2 = self.down[1](x1)
        
        latent = self.to_latent(x2)

        y = nn.functional.one_hot(y, num_classes=self.n_cls).type(torch.float)
        mask = mask.reshape(-1, 1).repeat(1, self.n_cls)
        mask = (1 - mask)     # due to the 1 is I want to mask so we have to flip 0->1 1->0
        y = y * mask
        
        label_emb1 = self.label_emb1(y).reshape(-1, 2 * self.feature, 1, 1)
        label_emb2 = self.label_emb2(y).reshape(-1, self.feature, 1, 1)
        time_emb1 = self.tim_emb1(t).reshape(-1, 2 * self.feature, 1, 1)
        time_emb2 = self.tim_emb2(t).reshape(-1, self.feature, 1, 1)

        x3 = self.up[0](latent)
        
        x4 = self.up[1](label_emb1 * x3 + time_emb1, x2)
        x5 = self.up[2](label_emb2 * x4 + time_emb2, x1)

        
        out = self.out(torch.cat([x, x5], dim=1))
        
        return out

# device = 'cuda'
# x = torch.randn((3, 3, 28, 28)).to(device)
# y = torch.ones((3), dtype=torch.long).to(device)
# t = torch.randn((3)).to(device)
# mask = torch.randn((3)).to(device)
# model = Conditional_Denoised_Unet().to(device)
# print(model(x, y, t, mask).shape)