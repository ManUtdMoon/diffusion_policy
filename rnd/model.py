import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



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
    }
    
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
