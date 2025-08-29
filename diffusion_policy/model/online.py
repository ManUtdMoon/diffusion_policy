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
            log_std_min=-20.0,
            log_std_max=2.0,
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
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.mean = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        self.log_std = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

        self.input_type = input_type

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        log_std = torch.tanh(self.log_std(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)

        return mean, log_std
    
    def get_eval_action(self, x):
        mean = self.mean(self.net(x))
        return torch.tanh(mean)

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t) # [-1, 1]

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True) # (B, 1)
        
        return {
            'sample': y_t,
            'mean': torch.tanh(mean),
            'log_prob': log_prob,
        }
