from typing import Optional

import numpy as np
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

from zprl.common.reactive_history import NoiseHistoryReplayBufferSamples


class ReactiveHistoryNoiseReplayBuffer(NStepReplayBuffer):
    def __init__(self, *args, exec_dim: int, action_step_dim: int,
                 history_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.optimize_memory_usage
        assert len(self.obs_shape) == 1
        assert history_length > 0 and action_step_dim > 0
        assert exec_dim > 0 and exec_dim % action_step_dim == 0
        self.exec_dim = exec_dim
        self.action_step_dim = action_step_dim
        self.max_exec_steps = exec_dim // action_step_dim
        self.history_length = history_length
        shape = (self.buffer_size, self.n_envs, exec_dim)
        self.a_exec = np.zeros(shape, dtype=np.float32)
        self.exec_steps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.episode_ids = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self.raw_timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def add(self, obs, next_obs, action, reward, done, infos, *,
            a_exec, exec_steps, episode_id):
        expected = (self.n_envs, self.exec_dim)
        a_exec = np.asarray(a_exec, dtype=np.float32)
        exec_steps = np.asarray(exec_steps, dtype=np.int64)
        episode_id = np.asarray(episode_id, dtype=np.int64)
        assert a_exec.shape == expected
        assert np.asarray(action).shape == (self.n_envs, self.action_dim)
        assert exec_steps.shape == episode_id.shape == (self.n_envs,)
        assert np.all((0 <= exec_steps) & (exec_steps <= self.max_exec_steps))
        assert np.all(episode_id >= 0)
        assert len(infos) == self.n_envs
        raw_timeouts = np.array([info.get('TimeLimit.truncated', False)
                                 for info in infos], dtype=bool)
        assert not np.any(raw_timeouts & ~np.asarray(done, dtype=bool))

        slot = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self.a_exec[slot] = a_exec
        self.exec_steps[slot] = exec_steps
        self.episode_ids[slot] = episode_id
        self.raw_timeouts[slot] = raw_timeouts

    def _history(self, index, env_index, endpoint_obs=None):
        length = self.history_length
        candidate = (index[:, None] - np.arange(length - 1, -1, -1)[None]
                     + (endpoint_obs is not None)) % self.buffer_size
        obs_valid = (self.episode_ids[candidate, env_index[:, None]]
                     == self.episode_ids[index, env_index][:, None])
        obs_valid[:, -1] = True
        # Stop at the first episode mismatch when looking backwards.
        obs_valid = np.logical_and.accumulate(obs_valid[:, ::-1], axis=1)[:, ::-1].copy()
        obs = self.observations[candidate, env_index[:, None]]
        a_exec = self.a_exec[candidate, env_index[:, None]]
        exec_steps = self.exec_steps[candidate, env_index[:, None]]
        obs[:, -1] = (self.observations[index, env_index] if endpoint_obs is None
                      else endpoint_obs)
        obs[~obs_valid] = 0
        a_exec[~obs_valid] = 0
        exec_steps[~obs_valid] = 0
        a_exec[:, -1] = 0
        exec_steps[:, -1] = 0
        return obs, a_exec, obs_valid, exec_steps

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> NoiseHistoryReplayBufferSamples:
        assert env is None, 'ReactiveHistoryNoiseReplayBuffer does not support VecNormalize'
        batch_inds = np.asarray(batch_inds, dtype=np.int64)
        env_indices = np.random.randint(0, self.n_envs, size=batch_inds.shape)
        assert np.all((0 <= batch_inds) & (batch_inds < self.buffer_size))
        if not self.full:
            assert np.all(batch_inds < self.pos)
        assert np.all(self.episode_ids[batch_inds, env_indices] >= 0)

        last_valid_index = (self.pos - 1) % self.buffer_size
        steps = np.arange(self.n_steps)[None]
        indices = (batch_inds[:, None] + steps) % self.buffer_size
        max_steps = (last_valid_index - batch_inds) % self.buffer_size + 1
        valid = (steps < max_steps[:, None]) & (
            self.episode_ids[indices, env_indices[:, None]]
            == self.episode_ids[batch_inds, env_indices][:, None])
        ended = np.logical_or(self.dones[indices, env_indices[:, None]],
                              self.raw_timeouts[indices, env_indices[:, None]])
        # Include the ending transition, but exclude everything after it.
        valid[:, 1:] &= np.logical_and.accumulate(valid & ~ended, axis=1)[:, :-1]
        lengths = valid.sum(axis=1)
        last_indices = (batch_inds + lengths - 1) % self.buffer_size
        rewards = np.where(valid, self.rewards[indices, env_indices[:, None]], 0)
        returns = (rewards * np.power(self.gamma, steps, dtype=np.float32)).sum(
            axis=1, keepdims=True)

        obs, a_exec, obs_valid, exec_steps = self._history(batch_inds, env_indices)
        next_obs, next_a_exec, next_obs_valid, next_exec_steps = self._history(
            last_indices, env_indices, self.next_observations[last_indices, env_indices])
        dones = self.dones[last_indices, env_indices] * (
            1 - self.raw_timeouts[last_indices, env_indices]
            if self.handle_timeout_termination else 1)
        discounts = np.power(self.gamma, lengths, dtype=np.float32)[:, None]
        return NoiseHistoryReplayBufferSamples(
            obs=self.to_torch(obs),
            a_exec=self.to_torch(a_exec),
            obs_valid=self.to_torch(obs_valid),
            exec_steps=self.to_torch(exec_steps),
            next_obs=self.to_torch(next_obs),
            next_a_exec=self.to_torch(next_a_exec),
            next_obs_valid=self.to_torch(next_obs_valid),
            next_exec_steps=self.to_torch(next_exec_steps),
            action=self.to_torch(self.actions[batch_inds, env_indices]),
            rewards=self.to_torch(returns),
            dones=self.to_torch(dones[:, None].astype(np.float32)),
            discounts=self.to_torch(discounts),
        )
