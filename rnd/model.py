import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


    
# Activation function mapping
activation_map = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'leaky_relu': nn.LeakyReLU(),
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),
    'sigmoid': nn.Sigmoid(),
    'identity': nn.Identity(),
    'mish': nn.Mish(),
    'none': nn.Identity(),
}


def create_mlp(
    input_dim: int,
    output_dim: int, 
    hidden_dims: list,
    activation_fn: str = 'relu',
    layer_norm: bool = False,
    activation_last: bool = False,
    dropout_prob: float = 0.0,
    bias: bool = True,
) -> nn.Sequential:
    """
    Create a Multi-Layer Perceptron (MLP) with flexible configuration.
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        hidden_dims (list): List of hidden layer dimensions
        activation_fn (str): Activation function name ('relu', 'tanh', 'elu', 'leaky_relu', 'gelu', 'silu')
        layer_norm (bool): Whether to use LayerNorm after each hidden layer
        batch_norm (bool): Whether to use BatchNorm1d after each hidden layer (ignored if layer_norm=True)
        activation_last (bool): Whether to apply activation function to the output layer
        dropout_prob (float): Dropout probability (0.0 means no dropout)
        bias (bool): Whether to use bias in linear layers
        output_activation (str): Specific activation for output layer (overrides activation_last)
    
    Returns:
        nn.Sequential: MLP model
    """  
    if activation_fn not in activation_map:
        raise ValueError(f"Unsupported activation function: {activation_fn}. "
                        f"Supported: {list(activation_map.keys())}")
    
    layers = []
    
    # Build hidden layers
    dims = [input_dim] + hidden_dims
    for i in range(len(dims) - 1):
        # Linear layer
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
        # Normalization
        if layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        # Activation
        layers.append(activation_map[activation_fn])
        # Dropout
        if dropout_prob > 0.0:
            layers.append(nn.Dropout(dropout_prob))
    
    # Output layer
    layers.append(nn.Linear(dims[-1], output_dim, bias=bias))
    if activation_last:
        layers.append(activation_map[activation_fn])
    
    return nn.Sequential(*layers)


class MLPFilmLayer(nn.Module):
    def __init__(self,
            input_dim: int,
            cond_dim: int,
            output_dim: int,
            activation_fn: str = 'relu'):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.cond_linear = nn.Linear(cond_dim, output_dim * 2)
        self.activation = activation_map.get(activation_fn, nn.Identity())
    
    def forward(self, x, cond):
        h = self.linear(x)  # (B, output_dim)
        cond_params = self.cond_linear(cond)  # (B, output_dim * 2)
        scale, bias = cond_params.chunk(2, dim=-1)
        h = h * scale + bias  # Apply FiLM modulation
        return self.activation(h)  # (B, output_dim)


class MLPFilm(nn.Module):
    def __init__(self,
            input_dim,
            cond_dim,
            hidden_dims,
            output_dim,
            activation_fn='relu'):
        super().__init__()
        
        self.layers = nn.ModuleList()
        this_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(
                MLPFilmLayer(this_dim, cond_dim, h_dim, activation_fn=activation_fn))
            this_dim = h_dim
        self.out_layer = nn.Linear(this_dim, output_dim)
    
    def forward(self, x, cond):
        for layer in self.layers:
            x = layer(x, cond)
        return self.out_layer(x)


class RND(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(RND, self).__init__()

        self.predictor = create_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation_fn='silu',
        )

        self.target = create_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation_fn='silu',
        )
        self.target.eval()
        self.target.requires_grad_(False)

        self.dist = nn.PairwiseDistance(p=2)

    def forward(self, x):
        pred_feat = self.predictor(x)  # (B, output_dim)
        target_feat = self.target(x)   # (B, output_dim)
        return self.dist(pred_feat, target_feat)  # (B,)
    
    def compute_loss(self, x):
        return self.forward(x).mean()


class RNDUnet(nn.Module):
    def __init__(self, input_dim, hidden_dims, cond_dim):
        super().__init__()

        self.predictor = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=cond_dim,
            diffusion_step_embed_dim=128,
            down_dims=hidden_dims,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=True
        )

        self.target = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=cond_dim,
            diffusion_step_embed_dim=128,
            down_dims=hidden_dims,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=True
        )

        self.target.eval()
        self.target.requires_grad_(False)

        self.dist = nn.PairwiseDistance(p=2)

    def forward(self, obs_emb, action):
        t = torch.zeros(action.shape[0], dtype=torch.long, device=action.device)  # Dummy diffusion step
        pred_feat = self.predictor(action, t, global_cond=obs_emb).flatten(1) # (B,H*Da)
        target_feat = self.target(action, t, global_cond=obs_emb).flatten(1) # (B,H*Da)
        return self.dist(pred_feat, target_feat)  # (B,)
    
    def compute_loss(self, obs_emb, action):
        return self.forward(obs_emb, action).mean()  # Average loss over batch


class LogZOMlp(nn.Module):
    def __init__(self, input_dim, hidden_dims, diffusion_step_dim: int = 64):
        super().__init__()
        dsd = diffusion_step_dim
        
        self.model = MLPFilm(
            input_dim=input_dim,
            cond_dim=dsd,
            hidden_dims=hidden_dims,
            output_dim=input_dim,
            activation_fn='mish'
        )

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsd),
            nn.Linear(dsd, dsd * 4),
            nn.Mish(),
            nn.Linear(dsd * 4, dsd),
        )

        self.time_scale = 100

    def forward(self, obs_emb, action):
        # compute logZO scores
        x = torch.cat([obs_emb, action.flatten(1)], dim=-1)  # (B, Do + H*Da)
        diffusion_step = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        cond = self.diffusion_step_encoder(diffusion_step)  # (B, dsd)
        v_pred = self.model(x, cond)

        zo = v_pred + x
        logZO = zo.reshape(x.shape[0], -1).pow(2).sum(dim=-1)# (B,)
        return logZO

    def compute_loss(self, obs_emb, action):
        # 1. build input and cond
        action = action.flatten(1)  # (B, H*Da)
        x0 = torch.cat([obs_emb, action], dim=-1)  # (B, Do + H*Da)
        x1 = torch.randn_like(x0)  # (B, Do + H*Da)
        bs = x0.shape[0]
        v_target = x1 - x0

        cont_t = torch.rand(bs, device=x0.device) # (B,)
        disc_t = (cont_t * self.time_scale).long()  # (B,)
        cont_t = cont_t.view(-1, *([1] * (x0.ndim - 1)))
        cond = self.diffusion_step_encoder(disc_t)  # (B, dsd)

        xt = x0 + cont_t * v_target

        # 2. compute vel
        v_pred = self.model(xt, cond)  # (B, Do + H*Da)
        loss = F.mse_loss(v_pred, v_target)

        return loss
