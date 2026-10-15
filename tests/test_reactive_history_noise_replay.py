import unittest
from unittest.mock import patch

import gymnasium
import numpy as np
import torch

from zprl.common.reactive_history import HistoryWindow, NoiseHistoryReplayBufferSamples
from zprl.common.reactive_history_noise_replay_buffer import ReactiveHistoryNoiseReplayBuffer
from tests.test_reactive_history_noise_policy import make_sum, make_obs


def make_buffer(n_steps=2, history_length=4, n_envs=1, capacity=8,
                bootstrap=True, noise_dim=2, exec_dim=4, obs_dim=1):
    return ReactiveHistoryNoiseReplayBuffer(
        capacity * n_envs,
        gymnasium.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
        gymnasium.spaces.Box(-1., 1., shape=(noise_dim,), dtype=np.float32),
        device='cpu', n_envs=n_envs, handle_timeout_termination=bootstrap,
        n_steps=n_steps, gamma=0.5, exec_dim=exec_dim, action_step_dim=1,
        history_length=history_length)


def sample_at(buffer, indices, env_indices=None):
    indices = np.asarray(indices, dtype=np.int64)
    env_indices = np.zeros_like(indices) if env_indices is None else np.asarray(env_indices)
    with patch('numpy.random.randint', return_value=env_indices):
        return buffer._get_samples(indices)


def add_row(buffer, time, episodes, done=None, timeout=None):
    n = buffer.n_envs
    done = np.zeros(n, dtype=bool) if done is None else np.asarray(done, dtype=bool)
    timeout = np.zeros(n, dtype=bool) if timeout is None else np.asarray(timeout, dtype=bool)
    obs = (time + np.arange(n) * 100).astype(np.float32)[:, None]
    reward = np.arange(n, dtype=np.float32) + time + 1
    action = np.full((n, buffer.action_dim), time / 100, dtype=np.float32)
    executed = np.repeat((obs + 10), buffer.exec_dim, axis=1)
    steps = np.where(done, 1, buffer.max_exec_steps).astype(np.int64)
    executed.reshape(n, buffer.max_exec_steps, -1)[
        np.arange(buffer.max_exec_steps)[None] >= steps[:, None]] = 0
    next_obs = np.where(timeout[:, None], 999., obs + 1).astype(np.float32)
    row = dict(obs=obs, next_obs=next_obs, action=action, reward=reward, done=done,
               infos=[{'TimeLimit.truncated': bool(t)} for t in timeout],
               a_exec=executed, exec_steps=steps, episode_id=np.asarray(episodes).copy())
    buffer.add(**row)
    return row


def chronological_reference(buffer, rows, anchor, env):
    """Use absolute time and retained records, independent of ring slot indexing."""
    oldest = max(0, len(rows) - buffer.buffer_size)
    episode = rows[anchor]['episode_id'][env]
    endpoint = anchor
    rewards = 0.
    length = 0
    for time in range(anchor, min(anchor + buffer.n_steps, len(rows))):
        row = rows[time]
        if row['episode_id'][env] != episode:
            break
        rewards += buffer.gamma ** length * row['reward'][env]
        length += 1
        endpoint = time
        if row['done'][env] or row['infos'][env]['TimeLimit.truncated']:
            break

    def history(time, next_state=False):
        obs = np.zeros((buffer.history_length, *buffer.obs_shape), dtype=np.float32)
        executed = np.zeros((buffer.history_length, buffer.exec_dim), dtype=np.float32)
        valid = np.zeros(buffer.history_length, dtype=bool)
        steps = np.zeros(buffer.history_length, dtype=np.int64)
        obs[-1] = rows[time]['next_obs' if next_state else 'obs'][env]
        valid[-1] = True
        for offset in range(1, buffer.history_length):
            past = time - offset + next_state
            if past < oldest or rows[past]['episode_id'][env] != rows[time]['episode_id'][env]:
                break
            obs[-1 - offset] = rows[past]['obs'][env]
            executed[-1 - offset] = rows[past]['a_exec'][env]
            steps[-1 - offset] = rows[past]['exec_steps'][env]
            valid[-1 - offset] = True
        return obs, executed, valid, steps

    current, following = history(anchor), history(endpoint, True)
    done = rows[endpoint]['done'][env]
    if buffer.handle_timeout_termination and rows[endpoint]['infos'][env]['TimeLimit.truncated']:
        done = False
    return NoiseHistoryReplayBufferSamples(*map(torch.as_tensor, (
        *current, *following, rows[anchor]['action'][env],
        np.array([rewards], dtype=np.float32), np.array([done], dtype=np.float32),
        np.array([buffer.gamma ** length], dtype=np.float32))))


class NoiseHistoryReplayTest(unittest.TestCase):
    def test_ring_and_episode_boundaries_match_chronological_reference(self):
        for n_steps in (1, 2, 9):
            for history_length in (1, 4, 12):
                for bootstrap in (False, True):
                    with self.subTest(n_steps=n_steps, history_length=history_length, bootstrap=bootstrap):
                        buffer = make_buffer(n_steps, history_length, n_envs=3, capacity=5, bootstrap=bootstrap)
                        rows = []
                        episodes = np.zeros(3, dtype=np.int64)
                        for time in range(17):
                            # All episodes are shorter than capacity, matching the experiment contract.
                            done = np.array([time % 3 == 2, time % 3 == 1, time % 4 == 2])
                            timeout = done & np.array([False, time % 2 == 0, True])
                            rows.append(add_row(buffer, time, episodes, done, timeout))
                            episodes += done
                            if time not in (0, 2, 4, 7, 16):
                                continue
                            raw = {key: value.copy() for key, value in vars(buffer).items()
                                   if isinstance(value, np.ndarray)}
                            times = list(range(max(0, time + 1 - 5), time + 1))
                            indices = [t % 5 for t in times for _ in range(3)]
                            envs = list(range(3)) * len(times)
                            actual = sample_at(buffer, indices, envs)
                            expected = [chronological_reference(buffer, rows, t, env)
                                        for t in times for env in range(3)]
                            for field in actual._fields:
                                wanted = torch.stack([getattr(row, field) for row in expected])
                                torch.testing.assert_close(getattr(actual, field), wanted)
                            for key, value in raw.items():
                                np.testing.assert_array_equal(getattr(buffer, key), value)

    def test_history_reconstruction_matches_existing_buffer(self):
        from zprl.common.reactive_history_replay_buffer import ReactiveHistoryReplayBuffer
        noise = make_buffer(n_steps=2, history_length=4, n_envs=2, capacity=8)
        old = ReactiveHistoryReplayBuffer(
            16, noise.observation_space, noise.action_space, device='cpu', n_envs=2,
            n_steps=2, gamma=0.5, base_action_dim=4, action_step_dim=1, history_length=4)
        episodes = np.zeros(2, dtype=np.int64)
        for time in range(15):
            done = np.array([time % 3 == 1, time % 4 == 2])
            row = add_row(noise, time, episodes, done, done & (time % 2 == 0))
            old.add(**row, a_base=row['a_exec'] + 1, next_a_base=row['a_exec'] + 2)
            episodes += done
        indices = np.repeat(np.arange(8), 2)
        envs = np.tile([0, 1], 8)
        expected = sample_at(old, indices, envs)
        actual = sample_at(noise, indices, envs)
        for field in actual._fields:
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field))

    def test_dual_dimensions_no_base_arrays_and_unfilled_slots(self):
        buffer = make_buffer(n_steps=9, history_length=12, noise_dim=2, exec_dim=4)
        add_row(buffer, 0, [0])
        buffer.observations[1:] = np.nan
        buffer.rewards[1:] = np.nan
        buffer.a_exec[1:] = np.nan
        batch = sample_at(buffer, [0])
        self.assertIsInstance(batch, NoiseHistoryReplayBufferSamples)
        self.assertEqual(batch.action.shape, (1, 2))
        self.assertEqual(batch.a_exec.shape, (1, 12, 4))
        self.assertFalse(hasattr(buffer, 'a_base'))
        self.assertFalse(hasattr(buffer, 'next_a_base'))
        self.assertNotIn('a_base', batch._fields)
        self.assertEqual(batch.obs_valid.sum(), 1)
        self.assertEqual(batch.next_obs_valid.sum(), 2)
        for value in batch:
            self.assertTrue(torch.isfinite(value).all())
        with self.assertRaises(AssertionError):
            sample_at(buffer, [1])

    def test_noise_can_be_wider_or_narrower_than_executed_action(self):
        for noise_dim, exec_dim in ((10, 40), (20, 10)):
            buffer = make_buffer(noise_dim=noise_dim, exec_dim=exec_dim)
            add_row(buffer, 0, [0])
            batch = sample_at(buffer, [0])
            self.assertEqual(batch.action.shape, (1, noise_dim))
            self.assertEqual(batch.a_exec.shape, (1, 4, exec_dim))

    def test_online_and_replay_windows_and_encoder_match(self):
        for n_steps in (1, 2, 4):
            for h in (1, 5):
                with self.subTest(n_steps=n_steps, history_length=h):
                    combined, _ = make_sum(history_length=h)
                    buffer = ReactiveHistoryNoiseReplayBuffer(
                        16, gymnasium.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                        gymnasium.spaces.Box(-1., 1., shape=(10,), dtype=np.float32), device='cpu',
                        n_steps=n_steps, gamma=0.5, exec_dim=40, action_step_dim=10, history_length=h)
                    combined.reset(make_obs([1]))
                    current, following = [], []
                    for time in range(4):
                        current.append(combined.history.get())
                        combined.predict_action(None, global_step=time, learning_start=2)
                        transition = combined.on_env_step(make_obs([time + 2]), [False],
                                                         [{'executed_action_steps': 4}])
                        buffer.add(**{key: value.numpy() for key, value in transition.items()},
                                   reward=np.array([1.], dtype=np.float32), done=np.array([False]),
                                   infos=[{}], episode_id=np.array([0]))
                        following.append(combined.history.get())
                    batch = sample_at(buffer, [0])
                    for names, window in ((('obs', 'a_exec', 'obs_valid', 'exec_steps'), current[0]),
                                           (('next_obs', 'next_a_exec', 'next_obs_valid', 'next_exec_steps'),
                                            following[min(n_steps, 4) - 1])):
                        for name, expected in zip(names, window):
                            torch.testing.assert_close(getattr(batch, name), expected)
                    encoder = combined.noise_policy.actor.encoder
                    torch.testing.assert_close(encoder(batch.next_obs, batch.next_a_exec, batch.next_obs_valid),
                                               encoder(*following[min(n_steps, 4) - 1][:3]))
                    combined.noise_policy.compute_critic_loss(batch)[0].backward()

    def test_timeout_endpoint_history_excludes_reset_and_unexecuted_tail(self):
        for bootstrap in (True, False):
            combined, _ = make_sum(history_length=4)
            buffer = ReactiveHistoryNoiseReplayBuffer(
                16, gymnasium.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                gymnasium.spaces.Box(-1., 1., shape=(10,), dtype=np.float32), device='cpu',
                n_envs=2, n_steps=4, gamma=0.5, exec_dim=40, action_step_dim=10,
                history_length=4, handle_timeout_termination=bootstrap)
            combined.reset(make_obs([1, 2]))
            combined.predict_action(None, global_step=0, learning_start=1)
            terminal = {key: value[0] for key, value in make_obs([99], [1]).items()}
            infos = [{'executed_action_steps': 1, 'TimeLimit.truncated': True,
                      'terminal_observation': terminal}, {'executed_action_steps': 4}]
            transition = combined.on_env_step(make_obs([10, 3]), [True, False], infos)
            buffer.add(**{key: value.numpy() for key, value in transition.items()},
                       reward=np.array([1., 2.]), done=np.array([True, False]), infos=infos,
                       episode_id=np.array([0, 0]))
            batch = sample_at(buffer, [0, 0], [0, 1])
            torch.testing.assert_close(batch.next_obs[0, -1], torch.tensor([99., 1.]))
            self.assertEqual(batch.dones[0].item(), 0 if bootstrap else 1)
            self.assertEqual(batch.next_exec_steps[0, -2].item(), 1)
            torch.testing.assert_close(batch.next_a_exec[0, -2, 10:], torch.zeros(30))
            online = combined.history.get()
            torch.testing.assert_close(online[0][0, -1], torch.tensor([10., 0.]))
            self.assertEqual(online[2][0].sum(), 1)
            for field, expected in zip(('next_obs', 'next_a_exec', 'next_obs_valid', 'next_exec_steps'), online):
                torch.testing.assert_close(getattr(batch, field)[1], expected[1])
            # Independently reconstruct the old episode terminal window.
            endpoint = HistoryWindow(4, 40)
            endpoint.reset(transition['obs'])
            endpoint.complete(transition['a_exec'], transition['exec_steps'])
            endpoint.append(transition['next_obs'])
            for field, expected in zip(('next_obs', 'next_a_exec', 'next_obs_valid', 'next_exec_steps'), endpoint.get()):
                torch.testing.assert_close(getattr(batch, field)[0], expected[0])


if __name__ == '__main__':
    unittest.main()
