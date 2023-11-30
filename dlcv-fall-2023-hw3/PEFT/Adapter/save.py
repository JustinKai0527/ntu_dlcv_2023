import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)                      # B, T, C  
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)          # B, H, T, C // H  
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)          # B, H, T, C // H
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)          # B, H, T, C // H
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))          # B, H, T, T
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))          # register_buffer bias up-triangle mask
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

# text_dim = (N, T, E)  img_dim = (N, S, feature_dim)
class CrossAttention(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # self.enc_attn = nn.Linear(cfg.n_embd, 2 * cfg.n_embd)
        # self.dec_attn = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.mha = nn.MultiheadAttention(embed_dim=cfg.n_embd, num_heads=cfg.n_head, batch_first=True)
    
    def forward(self, x, enc_feat):
        
        # k, v = self.enc_attn(enc_feat).split(self.cfg.n_embd, dim=2)
        # q = self.dec_attn(x)
        
        attn_out, attn_out_weights = self.mha(x, enc_feat, enc_feat)
        # attn_out (N, T, E)    attn_out_weights (N, T, S)
        return attn_out

class Adapter(nn.Module):
    def __init__(self, cfg):
        super(Adapter, self).__init__()
        
        self.down = nn.Linear(cfg.n_embd, 32)
        self.activation = nn.GELU()
        self.up = nn.Linear(32, cfg.n_embd)
    
    def forward(self, x):
        out = self.up(self.activation(self.down(x)))
        return out + x      # residual 

class Block(nn.Module):

    def __init__(self, cfg, adapter_exist=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_cross = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.cross_attn = CrossAttention(cfg)
        self.adater_exist = adapter_exist
        if adapter_exist:
            self.adapter_attn = Adapter(cfg)
            self.adapter_mlp = Adapter(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))

    def forward(self, x, enc_feat):
        x = x + self.attn(self.ln_1(x))
        if self.adater_exist:
            x = x + self.adapter_attn(self.cross_attn(self.ln_cross(x), enc_feat))
            x = x + self.adapter_mlp(self.mlp(self.ln_2(x)))
        else:
            x = x + self.cross_attn(self.ln_cross(x), enc_feat)
            x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) if i < (cfg.n_layer - 4) else Block(cfg, adapter_exist=True) for i in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, enc_feat: Tensor):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)   # word token embed, word pos embed
        # x = self.lm_head(self.transformer.ln_f(self.transformer.h(x)))
        for i in range(self.cfg.n_layer):
            x = self.transformer.h[i](x, enc_feat)
        x = self.lm_head(self.transformer.ln_f(x))
        return x
    
# cfg = Config("hw3_data/p2_data/decoder_model.bin")
# decoder = Decoder(cfg)
# state_dict = decoder.state_dict()
# # decoder.load_state_dict(torch.load("hw3_data/p2_data/decoder_model.bin"), strict=False)
# # state_dict = torch.load("hw3_data/p2_data/decoder_model.bin")
# for key in state_dict:
#     print(key)
