import unittest
from unittest.mock import patch

import gymnasium
import numpy as np
import torch

from zprl.common.reactive_history import HistoryReplayBufferSamples
from zprl.common.reactive_history_replay_buffer import ReactiveHistoryReplayBuffer
from zprl.common.reactive_replay_buffer import ReactiveNStepReplayBuffer


def make_buffer(n_steps=2, history_length=4, n_envs=1, buffer_size=16,
                handle_timeout_termination=True):
    obs_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
    action_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
    return ReactiveHistoryReplayBuffer(
        buffer_size, obs_space, action_space, device='cpu', n_envs=n_envs,
        handle_timeout_termination=handle_timeout_termination,
        n_steps=n_steps, gamma=0.5, base_action_dim=2,
        action_step_dim=1, history_length=history_length)


def add_step(buffer, obs, base, action, exec_action, reward, next_obs, next_base,
             episode_id, done=False, timeout=False, exec_steps=2):
    n_envs = buffer.n_envs

    def vector(value, dtype=np.float32):
        value = np.asarray(value, dtype=dtype)
        return np.full(n_envs, value, dtype=dtype) if value.ndim == 0 else value.reshape(n_envs)

    def pair(value):
        value = vector(value)
        return np.stack([value, value + 0.25], axis=-1)

    dones = vector(done, bool)
    timeouts = vector(timeout, bool)
    infos = [{'TimeLimit.truncated': bool(value)} for value in timeouts]
    buffer.add(
        obs=vector(obs)[:, None], next_obs=vector(next_obs)[:, None],
        action=pair(action), reward=vector(reward), done=dones, infos=infos,
        a_base=pair(base), next_a_base=pair(next_base), a_exec=pair(exec_action),
        exec_steps=vector(exec_steps, np.int64),
        episode_id=vector(episode_id, np.int64))


def sample_at(buffer, batch_inds, env_indices=None):
    batch_inds = np.asarray(batch_inds, dtype=np.int64)
    if env_indices is None:
        env_indices = np.zeros_like(batch_inds)
    with patch('stable_baselines3.common.buffers.np.random.randint',
               return_value=np.asarray(env_indices, dtype=np.int64)):
        return buffer._get_samples(batch_inds)


def scalar_samples(buffer, batch_inds, env_indices):
    # The pre-vectorization algorithm is an independent indexing oracle.
    def history(index, env_index, endpoint_obs=None):
        length = buffer.history_length
        obs = np.zeros((length, *buffer.obs_shape), dtype=np.float32)
        a_exec = np.zeros((length, buffer.base_action_dim), dtype=np.float32)
        obs_valid = np.zeros(length, dtype=bool)
        exec_steps = np.zeros(length, dtype=np.int64)
        obs[-1] = buffer.observations[index, env_index] if endpoint_obs is None else endpoint_obs
        obs_valid[-1] = True
        for offset in range(1, length):
            candidate = (index - offset + (endpoint_obs is not None)) % buffer.buffer_size
            if buffer.episode_ids[candidate, env_index] != buffer.episode_ids[index, env_index]:
                break
            position = length - 1 - offset
            obs[position] = buffer.observations[candidate, env_index]
            a_exec[position] = buffer.a_exec[candidate, env_index]
            obs_valid[position] = True
            exec_steps[position] = buffer.exec_steps[candidate, env_index]
        return obs, a_exec, obs_valid, exec_steps

    last_valid_index = (buffer.pos - 1) % buffer.buffer_size
    last_indices = np.empty(len(batch_inds), dtype=np.int64)
    lengths = np.empty(len(batch_inds), dtype=np.int64)
    returns = np.zeros((len(batch_inds), 1), dtype=np.float32)
    for row, (start, env_index) in enumerate(zip(batch_inds, env_indices)):
        max_steps = min(buffer.n_steps, (last_valid_index - start) % buffer.buffer_size + 1)
        for step in range(max_steps):
            index = (start + step) % buffer.buffer_size
            if buffer.episode_ids[index, env_index] != buffer.episode_ids[start, env_index]:
                break
            returns[row, 0] += buffer.gamma ** step * buffer.rewards[index, env_index]
            last_indices[row] = index
            lengths[row] = step + 1
            if buffer.dones[index, env_index] or buffer.raw_timeouts[index, env_index]:
                break
    current = [history(index, env_index) for index, env_index in zip(batch_inds, env_indices)]
    following = [history(index, env_index, buffer.next_observations[index, env_index])
                 for index, env_index in zip(last_indices, env_indices)]
    obs, a_exec, obs_valid, exec_steps = map(np.stack, zip(*current))
    next_obs, next_a_exec, next_obs_valid, next_exec_steps = map(np.stack, zip(*following))
    dones = buffer.dones[last_indices, env_indices] * (
        1 - buffer.raw_timeouts[last_indices, env_indices]
        if buffer.handle_timeout_termination else 1)
    return HistoryReplayBufferSamples(*map(torch.from_numpy, (
        obs, buffer.a_base[batch_inds, env_indices], a_exec, obs_valid, exec_steps,
        next_obs, buffer.next_a_base[last_indices, env_indices], next_a_exec,
        next_obs_valid, next_exec_steps, buffer.actions[batch_inds, env_indices],
        returns, dones[:, None].astype(np.float32),
        np.power(buffer.gamma, lengths, dtype=np.float32)[:, None])))


class ReactiveHistoryReplayTest(unittest.TestCase):
    def test_vectorized_samples_match_scalar_reference(self):
        rng = np.random.default_rng(1234)
        for n_steps in (1, 2, 7, 23):
            for history_length in (1, 4, 19):
                for n_envs in (1, 3):
                    for bootstrap in (False, True):
                        with self.subTest(n_steps=n_steps, history_length=history_length,
                                          n_envs=n_envs, bootstrap=bootstrap):
                            buffer = make_buffer(n_steps, history_length, n_envs,
                                                 16 * n_envs, bootstrap)
                            buffer.gamma = 0.9801
                            episode_id = np.zeros(n_envs, dtype=np.int64)
                            remaining = rng.integers(1, 6, size=n_envs)
                            for step in range(32):
                                remaining -= 1
                                done = remaining == 0
                                timeout = done & (rng.random(n_envs) < 0.5)
                                values = rng.standard_normal((7, n_envs)).astype(np.float32)
                                add_step(buffer, *values, episode_id, done, timeout,
                                         exec_steps=rng.integers(0, 3, size=n_envs))
                                episode_id += done
                                remaining[done] = rng.integers(1, 6, size=done.sum())
                                if step + 1 not in (3, 16, 23, 32):
                                    continue
                                size = buffer.buffer_size if buffer.full else buffer.pos
                                batch_inds = np.concatenate([
                                    np.repeat(np.arange(size), n_envs),
                                    np.full(n_envs, (buffer.pos - 1) % buffer.buffer_size)])
                                env_indices = np.concatenate([
                                    np.tile(np.arange(n_envs), size), np.arange(n_envs)])
                                raw = {key: value.copy() for key, value in vars(buffer).items()
                                       if isinstance(value, np.ndarray)}
                                expected = scalar_samples(buffer, batch_inds, env_indices)
                                actual = sample_at(buffer, batch_inds, env_indices)
                                for field in actual._fields:
                                    atol, rtol = (1e-6, 1e-5) if field == 'rewards' else (0, 0)
                                    torch.testing.assert_close(getattr(actual, field),
                                                               getattr(expected, field),
                                                               atol=atol, rtol=rtol)
                                for key, value in raw.items():
                                    np.testing.assert_array_equal(getattr(buffer, key), value)

    def test_unwritten_nan_slots_do_not_leak_into_samples(self):
        buffer = make_buffer(n_steps=7, history_length=9)
        add_step(buffer, 1, 2, 3, 4, 5, 6, 7, 0)
        buffer.rewards[1:] = np.nan
        buffer.observations[1:] = np.nan
        buffer.a_exec[1:] = np.nan
        batch = sample_at(buffer, [0])
        for value in batch:
            self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(batch.rewards, torch.tensor([[5.]]))
        self.assertEqual(batch.obs_valid.sum(), 1)
        self.assertEqual(batch.next_obs_valid.sum(), 2)

    def test_same_index_obs_exec_and_endpoint_fields(self):
        buffer = make_buffer()
        add_step(buffer, 0, 10, 100, 1000, 1, 1, 11, 0, exec_steps=1)
        add_step(buffer, 1, 11, 101, 1001, 2, 2, 12, 0)
        raw_exec = buffer.a_exec.copy()
        batch = sample_at(buffer, [0])
        self.assertIsInstance(batch, HistoryReplayBufferSamples)
        self.assertEqual(batch.obs.shape, (1, 4, 1))
        self.assertEqual(batch.a_exec.shape, (1, 4, 2))
        self.assertEqual(batch.action.shape, (1, 2))
        self.assertEqual(batch.obs_valid.dtype, torch.bool)
        self.assertEqual(batch.exec_steps.dtype, torch.int64)
        torch.testing.assert_close(batch.obs[0, :, 0], torch.tensor([0., 0., 0., 0.]))
        torch.testing.assert_close(batch.a_exec, torch.zeros(1, 4, 2))
        torch.testing.assert_close(batch.a_base[0], torch.tensor([10., 10.25]))
        torch.testing.assert_close(batch.action[0], torch.tensor([100., 100.25]))
        torch.testing.assert_close(batch.next_a_base[0], torch.tensor([12., 12.25]))
        torch.testing.assert_close(batch.next_obs[0, :, 0], torch.tensor([0., 0., 1., 2.]))
        torch.testing.assert_close(batch.next_a_exec[0, 1], torch.tensor([1000., 1000.25]))
        torch.testing.assert_close(batch.next_a_exec[0, 2], torch.tensor([1001., 1001.25]))
        torch.testing.assert_close(batch.next_a_exec[0, 3], torch.zeros(2))
        torch.testing.assert_close(batch.next_exec_steps[0], torch.tensor([0, 1, 2, 0]))
        torch.testing.assert_close(batch.next_obs_valid[0], torch.tensor([False, True, True, True]))
        np.testing.assert_array_equal(buffer.a_exec, raw_exec)
        self.assertEqual(buffer.exec_steps[0, 0], 1)

    def test_h_one_and_latest_write_can_sample_immediately(self):
        buffer = make_buffer(n_steps=4, history_length=1)
        self.assertTrue(np.all(buffer.episode_ids == -1))
        add_step(buffer, 5, 15, 25, 35, 3, 6, 16, 0)
        batch = sample_at(buffer, [0])
        torch.testing.assert_close(batch.obs[0, 0, 0], torch.tensor(5.))
        torch.testing.assert_close(batch.next_obs[0, 0, 0], torch.tensor(6.))
        torch.testing.assert_close(batch.a_exec, torch.zeros(1, 1, 2))
        torch.testing.assert_close(batch.next_a_exec, torch.zeros(1, 1, 2))
        torch.testing.assert_close(batch.rewards, torch.tensor([[3.]]))
        torch.testing.assert_close(batch.discounts, torch.tensor([[0.5]]))
        with self.assertRaises(AssertionError):
            sample_at(buffer, [1])

    def test_rl_action_dimension_is_independent_of_exec_dimension(self):
        obs_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        action_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        buffer = ReactiveHistoryReplayBuffer(
            8, obs_space, action_space, device='cpu', n_steps=1,
            base_action_dim=2, action_step_dim=1, history_length=2)
        buffer.add(
            np.array([[1.]], dtype=np.float32), np.array([[2.]], dtype=np.float32),
            np.array([[3.]], dtype=np.float32), np.array([4.], dtype=np.float32),
            np.array([False]), [{}], a_base=np.array([[5., 6.]]),
            next_a_base=np.array([[7., 8.]]), a_exec=np.array([[9., 10.]]),
            exec_steps=np.array([2]), episode_id=np.array([0]))
        batch = sample_at(buffer, [0])
        self.assertEqual(batch.action.shape, (1, 1))
        self.assertEqual(batch.a_base.shape, (1, 2))
        self.assertEqual(batch.a_exec.shape, (1, 2, 2))

    def test_timeout_terminal_history_and_partial_env_reset(self):
        for bootstrap, expected_done in ((True, 0.), (False, 1.)):
            with self.subTest(bootstrap=bootstrap):
                buffer = make_buffer(n_steps=4, n_envs=2, buffer_size=16,
                                     handle_timeout_termination=bootstrap)
                add_step(buffer, [0, 100], [10, 110], [20, 120], [30, 130],
                         [1, 10], [1, 101], [11, 111], [0, 0])
                add_step(buffer, [1, 101], [11, 111], [21, 121], [31, 131],
                         [2, 20], [999, 102], [9990, 112], [0, 0],
                         done=[True, False], timeout=[True, False], exec_steps=[1, 2])
                add_step(buffer, [200, 102], [210, 112], [220, 122], [230, 132],
                         [100, 30], [201, 103], [211, 113], [1, 0])
                batch = sample_at(buffer, [0, 1, 2, 2], [0, 0, 0, 1])
                torch.testing.assert_close(batch.rewards[:, 0],
                                           torch.tensor([2., 2., 100., 30.]))
                torch.testing.assert_close(batch.discounts[:, 0],
                                           torch.tensor([0.25, 0.5, 0.5, 0.5]))
                torch.testing.assert_close(batch.dones[:, 0],
                                           torch.tensor([expected_done, expected_done, 0., 0.]))
                torch.testing.assert_close(batch.next_obs[:2, -1, 0], torch.tensor([999., 999.]))
                torch.testing.assert_close(batch.next_a_base[:2, 0], torch.tensor([9990., 9990.]))
                torch.testing.assert_close(batch.next_obs[0, -3:, 0],
                                           torch.tensor([0., 1., 999.]))
                torch.testing.assert_close(batch.next_a_exec[0, -2],
                                           torch.tensor([31., 31.25]))
                self.assertEqual(batch.next_exec_steps[0, -2].item(), 1)
                torch.testing.assert_close(batch.obs_valid[2],
                                           torch.tensor([False, False, False, True]))
                torch.testing.assert_close(batch.obs[3, -3:, 0],
                                           torch.tensor([100., 101., 102.]))
                self.assertEqual(buffer.episode_ids[2, 0], 1)
                self.assertEqual(buffer.episode_ids[2, 1], 0)

    def test_n_longer_than_history_and_newest_boundary(self):
        buffer = make_buffer(n_steps=4, history_length=2)
        for t, reward in enumerate((1., 2., 4.)):
            add_step(buffer, t, 10 + t, 20 + t, 30 + t, reward,
                     t + 1, 11 + t, 0)
        batch = sample_at(buffer, [0, 2])
        torch.testing.assert_close(batch.rewards[:, 0], torch.tensor([3., 4.]))
        torch.testing.assert_close(batch.discounts[:, 0], torch.tensor([0.125, 0.5]))
        torch.testing.assert_close(batch.next_obs[0, :, 0], torch.tensor([2., 3.]))
        torch.testing.assert_close(batch.next_a_exec[0, 0], torch.tensor([32., 32.25]))
        torch.testing.assert_close(batch.next_a_base[0, 0], torch.tensor(13.))

    def test_wrap_and_overwritten_prefix_stop_at_episode_id(self):
        buffer = make_buffer(n_steps=4, history_length=5, buffer_size=5)
        for t, episode_id in enumerate((0, 0, 1, 1, 1, 2, 2)):
            add_step(buffer, t, 10 + t, 20 + t, 30 + t, 1,
                     t + 1, 11 + t, episode_id,
                     done=t in (1, 4))
        self.assertEqual(buffer.pos, 2)
        batch = sample_at(buffer, [0, 1, 2, 4])
        torch.testing.assert_close(batch.obs_valid[0],
                                   torch.tensor([False, False, False, False, True]))
        torch.testing.assert_close(batch.obs_valid[1],
                                   torch.tensor([False, False, False, True, True]))
        torch.testing.assert_close(batch.obs_valid[2],
                                   torch.tensor([False, False, False, False, True]))
        torch.testing.assert_close(batch.next_obs[3, -1, 0], torch.tensor(5.))
        torch.testing.assert_close(batch.next_a_base[3, 0], torch.tensor(15.))
        torch.testing.assert_close(batch.rewards[3], torch.tensor([1.]))
        torch.testing.assert_close(batch.dones[3], torch.tensor([1.]))

    def test_same_episode_history_crosses_physical_wrap(self):
        buffer = make_buffer(n_steps=2, history_length=4, buffer_size=5)
        for t, episode_id in enumerate((0, 0, 1, 1, 2, 2)):
            add_step(buffer, t, 10 + t, 20 + t, 30 + t, 1,
                     t + 1, 11 + t, episode_id, done=t in (1, 3))
        batch = sample_at(buffer, [0])
        torch.testing.assert_close(batch.obs_valid[0],
                                   torch.tensor([False, False, True, True]))
        torch.testing.assert_close(batch.obs[0, -2:, 0], torch.tensor([4., 5.]))
        torch.testing.assert_close(batch.a_exec[0, -2], torch.tensor([34., 34.25]))
        torch.testing.assert_close(batch.next_obs[0, -3:, 0],
                                   torch.tensor([4., 5., 6.]))

    def test_current_and_endpoint_actions_are_never_in_their_own_history(self):
        buffer = make_buffer(n_steps=1, history_length=4)
        for t in range(4):
            add_step(buffer, t, 10 + t, 20 + t, 30 + t, 1,
                     t + 1, 11 + t, 0)
        original = sample_at(buffer, [2])
        buffer.a_exec[2, 0] += 10
        buffer.a_exec[3, 0] += 20
        changed = sample_at(buffer, [2])
        torch.testing.assert_close(original.a_exec, changed.a_exec)
        torch.testing.assert_close(original.next_a_exec[0, -1], changed.next_a_exec[0, -1])
        later = sample_at(buffer, [3])
        self.assertGreater((later.a_exec[0, -2] - original.next_a_exec[0, -2]).abs().max(), 0)

    def test_differential_n_step_reference(self):
        for n_steps in (1, 2, 4):
            with self.subTest(n_steps=n_steps):
                buffer = make_buffer(n_steps=n_steps, n_envs=2, buffer_size=20)
                obs_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
                action_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
                reference = ReactiveNStepReplayBuffer(
                    20, obs_space, action_space, device='cpu', n_envs=2,
                    handle_timeout_termination=True, n_steps=n_steps,
                    gamma=0.5, base_action_dim=2)
                rows = [
                    ([0, 100], [10, 110], [20, 120], [1, 10], [1, 101], [11, 111], [False, False], [False, False], [0, 0]),
                    ([1, 101], [11, 111], [21, 121], [2, 20], [99, 102], [109, 112], [True, False], [True, False], [0, 0]),
                    ([200, 102], [210, 112], [220, 122], [3, 30], [201, 103], [211, 113], [False, False], [False, False], [1, 0]),
                ]
                for obs, base, action, reward, next_obs, next_base, done, timeout, episode_id in rows:
                    add_step(buffer, obs, base, action, action, reward,
                             next_obs, next_base, episode_id, done, timeout)
                    obs_array = np.stack([np.asarray(obs), np.asarray(base),
                                          np.asarray(base) + 0.25], axis=-1).astype(np.float32)
                    next_array = np.stack([np.asarray(next_obs), np.asarray(next_base),
                                           np.asarray(next_base) + 0.25], axis=-1).astype(np.float32)
                    action_array = np.stack([np.asarray(action),
                                             np.asarray(action) + 0.25], axis=-1).astype(np.float32)
                    infos = [{'TimeLimit.truncated': bool(value)} for value in timeout]
                    reference.add(obs_array, next_array, action_array,
                                  np.asarray(reward, dtype=np.float32),
                                  np.asarray(done, dtype=bool), infos)
                batch = sample_at(buffer, [0, 0, 1, 2], [0, 1, 0, 1])
                old = sample_at(reference, [0, 0, 1, 2], [0, 1, 0, 1])
                torch.testing.assert_close(batch.rewards, old.rewards)
                torch.testing.assert_close(batch.dones, old.dones)
                torch.testing.assert_close(batch.discounts, old.discounts)
                torch.testing.assert_close(batch.obs[:, -1], old.observations)
                torch.testing.assert_close(batch.next_obs[:, -1], old.next_observations)
                torch.testing.assert_close(batch.action, old.actions[:, :2])
                torch.testing.assert_close(batch.a_base, old.actions[:, 2:4])
                torch.testing.assert_close(batch.next_a_base, old.actions[:, 4:6])


if __name__ == '__main__':
    unittest.main()
