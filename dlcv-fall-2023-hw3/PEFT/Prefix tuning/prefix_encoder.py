import torch.nn as nn
import torch

class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        

class PrefixEncoder(nn.Module):
    def __init__(self, config, virtual_tokens, prefix_projection=False):
        super().__init__()
        
        self.cfg = config
        self.prefix_projection = prefix_projection
        vocab_size = 100
        n_layer = config.n_layer
        n_embd = config.n_embd
        if prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = nn.Embedding(vocab_size, n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 64),
                nn.Tanh(),
                nn.Linear(32, n_layer * 3 * n_embd)
            )
        else:
            self.embedding = nn.Embedding(virtual_tokens, n_layer * 3 * n_embd)
    
    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_token = self.embedding(prefix)
            prefix_token = self.mlp(prefix_token)
        else:
            prefix_token = self.embedding(prefix)
        
        return prefix_token

config = Config()
prefix_enc = PrefixEncoder(config, 20, True)
print(sum([p.numel() for n, p in prefix_enc.named_parameters()]))
# 976960