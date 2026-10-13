import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputGate(nn.Linear):
    """Gate a branch using the unnormalized residual stream: x + sigmoid(Wx-b)*y."""

    def __init__(self, d_model):
        super().__init__(d_model, d_model)

    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        nn.init.constant_(self.bias, -1.0)

    def forward(self, x, branch):
        return x + torch.sigmoid(F.linear(x, self.weight, self.bias)) * branch


class OutputGatedTransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                              batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attention_gate = OutputGate(d_model)
        self.ffn_gate = OutputGate(d_model)
        self.norm_first = True

    def forward(self, x, mask=None):
        normalized = self.norm1(x)
        attention = self.self_attn(normalized, normalized, normalized,
                                   attn_mask=mask, need_weights=False)[0]
        x = self.attention_gate(x, self.dropout1(attention))
        ffn = self.linear2(self.dropout(F.relu(self.linear1(self.norm2(x)))))
        return self.ffn_gate(x, self.dropout2(ffn))


class OutputGatedTransformerEncoder(nn.Module):
    """Finite-window encoder; no recurrent memory or final layer normalization."""

    def __init__(self, d_model=256, nhead=4, num_layers=2,
                 dim_feedforward=1024, dropout=0.0):
        super().__init__()
        assert num_layers > 0
        layer = OutputGatedTransformerBlock(d_model, nhead, dim_feedforward, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return F.tanh(self.final_norm(x))
