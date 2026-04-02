import numpy as np
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class TruncatedNormal(pyd.Normal):
    """A truncated normal distribution that clamps samples to [low, high]."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, sample_shape=torch.Size(), clip=None):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
        )
        self.mu = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(self, x):
        x = self.net(x)
        mu = torch.tanh(self.mu(x))
        return mu

    def get_eval_action(self, x):
        return self.forward(x)

    def sample(self, x, stddev, clip=None):
        mu = self.forward(x)
        dist = TruncatedNormal(mu, stddev)
        return dist.sample(clip=clip)
