from typing import Optional

import numpy as np
import torch
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class ReactiveNStepReplayBuffer(NStepReplayBuffer):
    """Keep endpoint base actions aligned with n-step next observations."""

    def __init__(self, *args, base_action_dim: int, **kwargs):
        super().__init__(*args, **kwargs)
        assert base_action_dim > 0
        assert len(self.obs_shape) == 1
        assert self.obs_shape[0] > base_action_dim
        assert self.action_dim == base_action_dim
        self.base_action_dim = base_action_dim

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        assert env is None, "ReactiveNStepReplayBuffer does not support VecNormalize"
        samples = super()._get_samples(batch_inds, env)
        obs, base_action = samples.observations.split(
            [samples.observations.shape[-1] - self.base_action_dim,
             self.base_action_dim], dim=-1)
        next_obs, next_base_action = samples.next_observations.split(
            [samples.next_observations.shape[-1] - self.base_action_dim,
             self.base_action_dim], dim=-1)
        actions = torch.cat(
            [samples.actions, base_action, next_base_action], dim=-1)
        return samples._replace(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
        )
