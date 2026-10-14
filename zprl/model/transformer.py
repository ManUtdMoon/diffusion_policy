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


class HistoryEncoder(nn.Module):
    def __init__(self, obs_dim, exec_dim, history_length, d_model=256, nhead=4,
                 num_layers=4, dim_feedforward=1024, dropout=0.1, causal_mask=False,
                 debug=False, sub_ratio=1):
        super().__init__()
        assert history_length > 0
        assert isinstance(sub_ratio, int) and sub_ratio > 0
        assert d_model % nhead == 0
        self.obs_dim = obs_dim
        self.exec_dim = exec_dim
        self.history_length = history_length
        self.sub_ratio = sub_ratio
        self.output_dim = d_model + obs_dim
        self.nhead = nhead
        self.causal_mask = causal_mask
        self.debug = debug
        self.projection = nn.Linear(obs_dim + exec_dim, d_model)
        self.position = nn.Embedding((history_length - 1) // sub_ratio + 1, d_model)
        self.transformer = OutputGatedTransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, obs, a_exec, obs_valid):
        batch_size, history_length, obs_dim = obs.shape
        assert history_length == self.history_length and obs_dim == self.obs_dim
        assert a_exec.shape == (batch_size, history_length, self.exec_dim)
        assert obs_valid.shape == (batch_size, history_length)
        assert obs_valid.dtype == torch.bool
        if self.debug:
            assert torch.all(obs_valid[:, -1])
        assert obs.device == a_exec.device == obs_valid.device
        assert obs.dtype == a_exec.dtype

        start = (history_length - 1) % self.sub_ratio
        obs, a_exec, obs_valid = (obs[:, start::self.sub_ratio],
                                  a_exec[:, start::self.sub_ratio],
                                  obs_valid[:, start::self.sub_ratio])
        history_length = obs.shape[1]
        # The action for the decision being evaluated is never part of its history.
        a_exec = torch.cat([a_exec[:, :-1], torch.zeros_like(a_exec[:, -1:])], dim=1)
        obs = torch.where(obs_valid.unsqueeze(-1), obs, torch.zeros_like(obs))
        a_exec = torch.where(obs_valid.unsqueeze(-1), a_exec, torch.zeros_like(a_exec))
        x = self.projection(torch.cat([obs, a_exec], dim=-1))
        positions = torch.arange(history_length, device=obs.device)
        x = x + self.position(positions).unsqueeze(0)

        # True means blocked. Padded queries may attend valid keys so no row is
        # fully masked, including when causal attention meets left padding.
        mask = (~obs_valid)[:, None, :].expand(batch_size, history_length, history_length)
        if self.causal_mask:
            future = torch.ones(history_length, history_length, device=obs.device,
                                dtype=torch.bool).triu(1)
            mask = mask | (obs_valid[:, :, None] & future)
        mask = mask.repeat_interleave(self.nhead, dim=0)
        h = self.transformer(x, mask=mask)[:, -1]
        return torch.cat([h, obs[:, -1]], dim=-1)
