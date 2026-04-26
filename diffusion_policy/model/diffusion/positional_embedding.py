import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FourierTimeEmbedding(nn.Module):
    def __init__(self, dim, min_period=4e-3, max_period=4.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim ({dim}) must be divisible by 2")
        fraction = torch.linspace(0.0, 1.0, dim // 2)
        period = min_period * (max_period / min_period) ** fraction
        self.register_buffer('inv_period', 1.0 / period)

    def forward(self, x):
        x = x.float()
        emb = x[:, None] * self.inv_period[None, :] * (2 * math.pi)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
