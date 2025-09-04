import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(SoftQNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, 1), std=0.01)
        )

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self,
            obs_dim,
            action_dim,
            hidden_dim=256,
            input_type='obs_action', # 'obs' or 'obs_action'
        ):
        super().__init__()

        input_dim = None
        if input_type == 'obs':
            input_dim = obs_dim
        elif input_type == 'obs_action':
            input_dim = obs_dim + action_dim
        else:
            raise ValueError("input_type must be 'obs' or 'obs_action'")

        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self,
            obs_dim,
            action_dim,
            hidden_dim=256,
            input_type='obs', # 'obs' or 'obs_action'
            ):
        super(Actor, self).__init__()

        input_dim = None
        if input_type == 'obs':
            input_dim = obs_dim
        elif input_type == 'obs_action':
            input_dim = obs_dim + action_dim
        else:
            raise ValueError("input_type must be 'obs' or 'obs_action'")

        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.GELU(),
        )

        self.alpha_layer = layer_init(nn.Linear(hidden_dim, action_dim))
        self.beta_layer = layer_init(nn.Linear(hidden_dim, action_dim))        
        self.softplus = nn.Softplus()

        self.input_type = input_type

    def forward(self, x):
        x = self.net(x)
        epsilon = 1. + 1e-6  # do not want ill-conditioned beta distribution
        alpha = self.softplus(self.alpha_layer(x)) + epsilon
        beta = self.softplus(self.beta_layer(x)) + epsilon
        return alpha, beta

    def get_eval_action(self, x):
        alpha, beta = self.forward(x)
        dist = torch.distributions.Beta(alpha, beta)
        mean = dist.mode
        return 2. * mean - 1.

    def get_action(self, x):
        alpha, beta = self.forward(x)
        dist = torch.distributions.Beta(alpha, beta)

        x_t = dist.rsample()
        
        # First, clamp to get a definitely safe value
        clamped_x_t = self._clamp(x_t, low=0.0, high=1.0)
        
        # Use the safe value to calculate the final action
        y_t = 2. * clamped_x_t - 1.
        
        # For the 'mean' key, we can use the safe mode as well for consistency
        mean = 2. * dist.mode - 1.
        
        # log_prob must also use the safe value
        log_prob = (dist.log_prob(clamped_x_t) + torch.log(torch.tensor(0.5, device=clamped_x_t.device))).sum(dim=-1, keepdim=True) # (B, 1)

        assert torch.all(torch.isfinite(y_t))
        assert torch.all(torch.isfinite(log_prob))

        return {
            'sample': y_t,
            'mean': mean,
            'log_prob': log_prob,
        }

    def log_prob_action(self, x, action):
        alpha, beta = self.forward(x)
        dist = torch.distributions.Beta(alpha, beta)

        x_t = (action + 1.) / 2.
        # Clamp the value to be slightly away from the boundaries 0 and 1
        clamped_x_t = self._clamp(x_t, low=0.0, high=1.0)
        
        # Use the clamped value for log_prob calculation
        log_prob = (dist.log_prob(clamped_x_t) + torch.log(torch.tensor(0.5, device=clamped_x_t.device))).sum(dim=-1, keepdim=True) # (B, 1)
        return log_prob

    def _clamp(self, x, low=-1.0, high=1.0, eps=1e-6):
        clamped_x = torch.clamp(x, low + eps, high - eps)
        x = x - x.detach() + clamped_x.detach()
        return x
