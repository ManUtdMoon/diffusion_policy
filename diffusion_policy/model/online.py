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
        logstd = self._clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mean, logstd

    def get_eval_action(self, x):
        mean = self.mean(self.net(x))
        return torch.tanh(mean)

    def get_action(self, x):
        mean, logstd = self.forward(x)
        dist = torch.distributions.Normal(mean, logstd.exp())

        x_t = dist.rsample()
        y_t = torch.tanh(x_t) # [-1, 1]
        
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t**2 + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True) # (B, 1)

        assert torch.all(torch.isfinite(y_t))
        assert torch.all(torch.isfinite(log_prob))

        return {
            'sample': y_t,
            'mean': torch.tanh(mean),
            'log_prob': log_prob,
        }

    def log_prob_action(self, x, action):
        return None

    def _clamp(self, x, low=-1.0, high=1.0, eps=1e-6):
        clamped_x = torch.clamp(x, low + eps, high - eps)
        x = x - x.detach() + clamped_x.detach()
        return x

class BatchedLinear(nn.Module):
    def __init__(self, num_batch, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_batch, in_features, out_features))
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(num_batch, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Simplified Kaiming uniform initialization for the batch
        for i in range(self.weight.size(0)):
            nn.init.kaiming_uniform_(self.weight[i], a=5**0.5)
            if self.use_bias:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x):
        # x: (num_batch, B, in_features)
        x = torch.bmm(x, self.weight)
        if self.use_bias:
            x = x + self.bias
        return x

class BatchedSoftQNet(nn.Module):
    def __init__(self, obs_dim, action_dim, num_qs, hidden_dim=256):
        super().__init__()
        self.num_qs = num_qs
        self.net = nn.Sequential(
            BatchedLinear(num_qs, obs_dim + action_dim, hidden_dim),
            nn.GELU(),
            BatchedLinear(num_qs, hidden_dim, hidden_dim),
            nn.GELU(),
            BatchedLinear(num_qs, hidden_dim, hidden_dim),
            nn.GELU(),
            BatchedLinear(num_qs, hidden_dim, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        # expand for batch
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1) # (num_qs, B, D_in)
        q_logits = self.net(x) # (num_qs, B, 1)
        q_values = 0.5 * (torch.tanh(q_logits) + 1.)  # scale to [0, 1]
        return q_values