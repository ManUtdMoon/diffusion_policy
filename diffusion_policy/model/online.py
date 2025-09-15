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
            nn.Tanh(),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        # expand for batch
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1) # (num_qs, B, D_in)
        q_logits = self.net(x) # (num_qs, B, 1)
        q_values = 0.5 * (q_logits + 1.) # scale to [0, 1]
        return q_values