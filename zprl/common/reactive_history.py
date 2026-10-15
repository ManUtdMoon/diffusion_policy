from typing import NamedTuple, Optional

import torch


class HistoryReplayBufferSamples(NamedTuple):
    obs: torch.Tensor
    a_base: torch.Tensor
    a_exec: torch.Tensor
    obs_valid: torch.Tensor
    exec_steps: torch.Tensor
    next_obs: torch.Tensor
    next_a_base: torch.Tensor
    next_a_exec: torch.Tensor
    next_obs_valid: torch.Tensor
    next_exec_steps: torch.Tensor
    action: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    discounts: Optional[torch.Tensor]


class NoiseHistoryReplayBufferSamples(NamedTuple):
    obs: torch.Tensor
    a_exec: torch.Tensor
    obs_valid: torch.Tensor
    exec_steps: torch.Tensor
    next_obs: torch.Tensor
    next_a_exec: torch.Tensor
    next_obs_valid: torch.Tensor
    next_exec_steps: torch.Tensor
    action: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    discounts: Optional[torch.Tensor]


class HistoryWindow:
    def __init__(self, length: int, exec_dim: int):
        assert length > 0 and exec_dim > 0
        self.length = length
        self.exec_dim = exec_dim
        self.obs = None
        self.a_exec = None
        self.obs_valid = None
        self.exec_steps = None

    def reset(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        assert obs.ndim == 2
        batch_size, obs_dim = obs.shape
        if mask is None:
            self.obs = obs.new_zeros(batch_size, self.length, obs_dim)
            self.a_exec = obs.new_zeros(batch_size, self.length, self.exec_dim)
            self.obs_valid = torch.zeros(batch_size, self.length, device=obs.device,
                                         dtype=torch.bool)
            self.exec_steps = torch.zeros(batch_size, self.length, device=obs.device,
                                          dtype=torch.long)
            mask = torch.ones(batch_size, device=obs.device, dtype=torch.bool)
        else:
            assert self.obs is not None and self.obs.shape == (batch_size, self.length, obs_dim)
            assert mask.shape == (batch_size,) and mask.dtype == torch.bool
        self.obs[mask] = 0
        self.a_exec[mask] = 0
        self.obs_valid[mask] = False
        self.exec_steps[mask] = 0
        self.obs[mask, -1] = obs[mask]
        self.obs_valid[mask, -1] = True

    def complete(self, a_exec: torch.Tensor, exec_steps: torch.Tensor):
        assert self.obs is not None
        assert a_exec.shape == (self.obs.shape[0], self.exec_dim)
        assert exec_steps.shape == (self.obs.shape[0],)
        assert exec_steps.dtype == torch.long
        self.a_exec[:, -1] = a_exec
        self.exec_steps[:, -1] = exec_steps

    def append(self, obs: torch.Tensor):
        assert self.obs is not None and obs.shape == self.obs[:, -1].shape
        bs = obs.shape[0]
        self.obs = torch.cat([self.obs[:, 1:], obs[:, None]], dim=1)
        self.a_exec = torch.cat([self.a_exec[:, 1:], self.a_exec.new_zeros(bs, 1, self.exec_dim)], dim=1)
        self.obs_valid = torch.cat([self.obs_valid[:, 1:], self.obs_valid.new_ones(bs, 1)], dim=1)
        self.exec_steps = torch.cat([self.exec_steps[:, 1:], self.exec_steps.new_zeros(bs, 1)], dim=1)

    def get(self):
        assert self.obs is not None
        return (
            self.obs.clone(),
            self.a_exec.clone(),
            self.obs_valid.clone(),
            self.exec_steps.clone()
        )
