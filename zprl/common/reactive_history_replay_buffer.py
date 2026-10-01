from typing import Optional

import numpy as np
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

from zprl.common.reactive_history import HistoryReplayBufferSamples


class ReactiveHistoryReplayBuffer(NStepReplayBuffer):
    def __init__(self, *args, base_action_dim: int, action_step_dim: int,
                 history_length: int, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.optimize_memory_usage
        assert len(self.obs_shape) == 1
        assert history_length > 0 and action_step_dim > 0
        assert base_action_dim > 0 and base_action_dim % action_step_dim == 0
        self.base_action_dim = base_action_dim
        self.action_step_dim = action_step_dim
        self.max_exec_steps = base_action_dim // action_step_dim
        self.history_length = history_length
        shape = (self.buffer_size, self.n_envs, base_action_dim)
        self.a_base = np.zeros(shape, dtype=np.float32)
        self.next_a_base = np.zeros(shape, dtype=np.float32)
        self.a_exec = np.zeros(shape, dtype=np.float32)
        self.exec_steps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.episode_ids = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self.raw_timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=bool)

    def add(self, obs, next_obs, action, reward, done, infos, *,
            a_base, next_a_base, a_exec, exec_steps, episode_id):
        expected = (self.n_envs, self.base_action_dim)
        a_base = np.asarray(a_base, dtype=np.float32)
        next_a_base = np.asarray(next_a_base, dtype=np.float32)
        a_exec = np.asarray(a_exec, dtype=np.float32)
        exec_steps = np.asarray(exec_steps, dtype=np.int64)
        episode_id = np.asarray(episode_id, dtype=np.int64)
        assert a_base.shape == next_a_base.shape == a_exec.shape == expected
        assert exec_steps.shape == episode_id.shape == (self.n_envs,)
        assert np.all((0 <= exec_steps) & (exec_steps <= self.max_exec_steps))
        assert np.all(episode_id >= 0)
        assert len(infos) == self.n_envs
        raw_timeouts = np.array([info.get('TimeLimit.truncated', False)
                                 for info in infos], dtype=bool)
        assert not np.any(raw_timeouts & ~np.asarray(done, dtype=bool))

        slot = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self.a_base[slot] = a_base
        self.next_a_base[slot] = next_a_base
        self.a_exec[slot] = a_exec
        self.exec_steps[slot] = exec_steps
        self.episode_ids[slot] = episode_id
        self.raw_timeouts[slot] = raw_timeouts

    def _history(self, index, env_index, endpoint_obs=None):
        length = self.history_length
        obs = np.zeros((length, *self.obs_shape), dtype=self.observations.dtype)
        a_exec = np.zeros((length, self.base_action_dim), dtype=np.float32)
        obs_valid = np.zeros(length, dtype=bool)
        exec_steps = np.zeros(length, dtype=np.int64)
        obs[-1] = (self.observations[index, env_index] if endpoint_obs is None
                   else endpoint_obs)
        obs_valid[-1] = True
        episode_id = self.episode_ids[index, env_index]
        for offset in range(1, length):
            candidate = (index - offset + (endpoint_obs is not None)) % self.buffer_size
            if self.episode_ids[candidate, env_index] != episode_id:
                break
            position = length - 1 - offset
            obs[position] = self.observations[candidate, env_index]
            a_exec[position] = self.a_exec[candidate, env_index]
            exec_steps[position] = self.exec_steps[candidate, env_index]
            obs_valid[position] = True
        return obs, a_exec, obs_valid, exec_steps

    def _get_samples(self, batch_inds: np.ndarray,
                     env: Optional[VecNormalize] = None) -> HistoryReplayBufferSamples:
        assert env is None, 'ReactiveHistoryReplayBuffer does not support VecNormalize'
        batch_inds = np.asarray(batch_inds, dtype=np.int64)
        env_indices = np.random.randint(0, self.n_envs, size=batch_inds.shape)
        batch_size = len(batch_inds)
        assert np.all((0 <= batch_inds) & (batch_inds < self.buffer_size))
        if not self.full:
            assert np.all(batch_inds < self.pos)
        assert np.all(self.episode_ids[batch_inds, env_indices] >= 0)

        last_valid_index = (self.pos - 1) % self.buffer_size
        last_indices = np.empty(batch_size, dtype=np.int64)
        lengths = np.empty(batch_size, dtype=np.int64)
        returns = np.zeros((batch_size, 1), dtype=np.float32)
        for row, (start, env_index) in enumerate(zip(batch_inds, env_indices)):
            max_steps = min(self.n_steps,
                            (last_valid_index - start) % self.buffer_size + 1)
            episode_id = self.episode_ids[start, env_index]
            for step in range(max_steps):
                index = (start + step) % self.buffer_size
                if self.episode_ids[index, env_index] != episode_id:
                    break
                returns[row, 0] += self.gamma ** step * self.rewards[index, env_index]
                last_indices[row] = index
                lengths[row] = step + 1
                if self.dones[index, env_index] or self.raw_timeouts[index, env_index]:
                    break

        current = [self._history(index, env_index)
                   for index, env_index in zip(batch_inds, env_indices)]
        next_history = [self._history(index, env_index,
                                      self.next_observations[index, env_index])
                        for index, env_index in zip(last_indices, env_indices)]
        obs, a_exec, obs_valid, exec_steps = map(np.stack, zip(*current))
        next_obs, next_a_exec, next_obs_valid, next_exec_steps = map(
            np.stack, zip(*next_history))
        dones = self.dones[last_indices, env_indices] * (
            1 - self.raw_timeouts[last_indices, env_indices]
            if self.handle_timeout_termination else 1)
        discounts = np.power(self.gamma, lengths, dtype=np.float32)[:, None]
        return HistoryReplayBufferSamples(
            obs=self.to_torch(obs),
            a_base=self.to_torch(self.a_base[batch_inds, env_indices]),
            a_exec=self.to_torch(a_exec),
            obs_valid=self.to_torch(obs_valid),
            exec_steps=self.to_torch(exec_steps),
            next_obs=self.to_torch(next_obs),
            next_a_base=self.to_torch(self.next_a_base[last_indices, env_indices]),
            next_a_exec=self.to_torch(next_a_exec),
            next_obs_valid=self.to_torch(next_obs_valid),
            next_exec_steps=self.to_torch(next_exec_steps),
            action=self.to_torch(self.actions[batch_inds, env_indices]),
            rewards=self.to_torch(returns),
            dones=self.to_torch(dones[:, None].astype(np.float32)),
            discounts=self.to_torch(discounts),
        )
