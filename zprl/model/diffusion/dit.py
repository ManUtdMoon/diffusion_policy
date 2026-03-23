# Heavy inspiration taken from dit-policy by Sudeep Dasari: https://github.com/SudeepDasari/dit-policy

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 1, f"assumes 1d input timestep array, got {x.shape}"
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return self.out_net(x)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim, d_cond):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(d_cond, dim)
        self.shift = nn.Linear(d_cond, dim)

    def forward(self, x, c):
        # TODO: 1 + scale?
        c = self.act(c)
        return _modulate(x, self.scale(c), self.shift(c))

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim, d_cond):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(d_cond, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c).unsqueeze(1)

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(
        self, d_model, d_cond, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # TODO: no affine?
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        # TODO: remove dropout?
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # create modulation layers
        self.attn_mod1 = _ShiftScaleMod(d_model, d_cond)
        self.attn_mod2 = _ZeroScaleMod(d_model, d_cond)
        self.mlp_mod1 = _ShiftScaleMod(d_model, d_cond)
        self.mlp_mod2 = _ZeroScaleMod(d_model, d_cond)

    def forward(self, x, cond):
        x2 = self.attn_mod1(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), cond) + x

        x2 = self.mlp_mod1(self.norm2(x), cond)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), cond)
        return x + x2

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, cond_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, out_size, bias=True)

        self.reset_parameters()

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        # TODO: norm_final; 1 + scale ✅
        x = _modulate(self.norm_final(x), scale, shift)
        x = self.linear(x)
        return x

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerDecoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )

        for l in self.layers:
            l.reset_parameters()

    def forward(self, src, cond):
        x = src
        for layer in self.layers:
            x = layer(x, cond)
        return x


class DiTNoiseNet(nn.Module):
    def __init__(
        self,
        input_dim,  # da
        input_length,  # H
        cond_dim=512, # To * do
        time_dim=256,
        hidden_dim=512,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
    ):
        super().__init__()
        
        # pos emb
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(1, input_length, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # input encode
        self.time_net = _TimeNetwork(time_dim, time_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, hidden_dim),
        )

        # decoder blocks
        cond_dim = time_dim + cond_dim
        decoder_module = _DiTDecoder(
            hidden_dim,
            d_cond=cond_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # turns predicted tokens into epsilons
        self.final_layer = _FinalLayer(hidden_dim, cond_dim, input_dim)

        logger.info(
            f"number of parameters: {(sum(p.numel() for p in self.parameters()) / 1e6):.2f} M", 
        )
    
    def forward(self,
            sample: torch.Tensor,
            timestep: torch.Tensor,
            cond: torch.Tensor):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_net(timesteps)  # (B, time_dim)

        cond = torch.cat(
            [time_emb, cond], dim=-1
        )

        x = self.ac_proj(sample) + self.dec_pos # (B, H, hidden_dim)
        x = self.decoder(x, cond) # (B, H, hidden_dim)
        x = self.final_layer(x, cond) # (B, H, input_dim)

        return x


if __name__ == "__main__":
    model = DiTNoiseNet(
        input_dim=10,
        input_length=16,
        time_dim=128,
        hidden_dim=384,
        num_blocks=6,
    )

    bs = 2
    sample = torch.randn(bs, 16, 10)
    timestep = torch.randint(0, 1000, (bs,))
    cond = torch.randn(bs, 512)

    out = model(sample, timestep, cond)
    print(out.shape)
