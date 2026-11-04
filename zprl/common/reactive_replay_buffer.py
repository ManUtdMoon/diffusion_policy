from typing import Optional

import numpy as np
import torch
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class ReactiveNStepReplayBuffer(NStepReplayBuffer):
    """Keep endpoint base banks aligned with n-step next observations.

    base_action_dim is one chunk's width D. Storage appends K*D values to the
    state: selected base plus padding in obs, and the full bank in next_obs.
    Sampled actions contain [residual(D), selected_base(D), endpoint_bank(K*D)].
    """

    def __init__(self, *args, base_action_dim: int, num_base_candidates: int = 1, **kwargs):
        if (not isinstance(num_base_candidates, int) or isinstance(num_base_candidates, bool)
                or num_base_candidates < 1):
            raise ValueError('num_base_candidates must be a positive integer')
        super().__init__(*args, **kwargs)
        assert base_action_dim > 0
        assert len(self.obs_shape) == 1
        self.num_base_candidates = num_base_candidates
        self.bank_dim = num_base_candidates * base_action_dim
        assert self.obs_shape[0] > self.bank_dim
        assert not self.optimize_memory_usage
        assert self.action_dim == base_action_dim
        self.base_action_dim = base_action_dim
        self.state_dim = self.obs_shape[0] - self.bank_dim

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        assert env is None, "ReactiveNStepReplayBuffer does not support VecNormalize"
        samples = super()._get_samples(batch_inds, env)
        # Parent sampling selects the n-step endpoint before state/bank splitting.
        obs, base_action, _ = samples.observations.split(
            [self.state_dim, self.base_action_dim, self.bank_dim - self.base_action_dim], dim=-1)
        next_obs, next_base_action = samples.next_observations.split(
            [self.state_dim, self.bank_dim], dim=-1)
        actions = torch.cat(
            [samples.actions, base_action, next_base_action], dim=-1)
        return samples._replace(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
        )
