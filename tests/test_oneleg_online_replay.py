"""Fixed-trajectory oracle, runnable with both SB3 2.4.1 and 2.7.0."""

import os
import unittest
from unittest.mock import patch

import gymnasium
import numpy as np

from zprl.common.reactive_history_replay_buffer import ReactiveHistoryReplayBuffer
from zprl.common.reactive_history_noise_replay_buffer import ReactiveHistoryNoiseReplayBuffer


def scalar_sample(buffer, start, env_index):
    def history(index, endpoint=False):
        obs = np.zeros((buffer.history_length, 2), dtype=np.float32)
        actions = np.zeros((buffer.history_length, 4), dtype=np.float32)
        valid = np.zeros(buffer.history_length, dtype=bool)
        executed = np.zeros(buffer.history_length, dtype=np.int64)
        obs[-1] = (buffer.next_observations if endpoint else buffer.observations)[index, env_index]
        valid[-1] = True
        for offset in range(1, buffer.history_length):
            previous = (index - offset + int(endpoint)) % buffer.buffer_size
            if buffer.episode_ids[previous, env_index] != buffer.episode_ids[index, env_index]:
                break
            obs[-offset - 1] = buffer.observations[previous, env_index]
            actions[-offset - 1] = buffer.a_exec[previous, env_index]
            valid[-offset - 1] = True
            executed[-offset - 1] = buffer.exec_steps[previous, env_index]
        return obs, actions, valid, executed

    reward = 0.
    length = 0
    endpoint = start
    available = ((buffer.pos - 1 - start) % buffer.buffer_size) + 1
    for step in range(min(buffer.n_steps, available)):
        index = (start + step) % buffer.buffer_size
        if buffer.episode_ids[index, env_index] != buffer.episode_ids[start, env_index]:
            break
        reward += buffer.gamma ** step * buffer.rewards[index, env_index]
        length += 1
        endpoint = index
        if buffer.dones[index, env_index] or buffer.raw_timeouts[index, env_index]:
            break
    obs, actions, valid, executed = history(start)
    next_obs, next_actions, next_valid, next_executed = history(endpoint, True)
    done = buffer.dones[endpoint, env_index]
    if buffer.handle_timeout_termination and buffer.raw_timeouts[endpoint, env_index]:
        done = 0.
    result = dict(obs=obs, a_exec=actions, obs_valid=valid, exec_steps=executed,
        next_obs=next_obs, next_a_exec=next_actions, next_obs_valid=next_valid,
        next_exec_steps=next_executed, action=buffer.actions[start, env_index],
        rewards=np.array([reward], dtype=np.float32),
        dones=np.array([done], dtype=np.float32),
        discounts=np.array([buffer.gamma ** length], dtype=np.float32))
    if isinstance(buffer, ReactiveHistoryReplayBuffer):
        result.update(a_base=buffer.a_base[start, env_index],
            next_a_base=buffer.next_a_base[endpoint, env_index])
    return result


class OneLegOnlineReplayTest(unittest.TestCase):
    def test_fixed_trajectories_and_public_sampling(self):
        fixture = {}
        for method, cls in [('res', ReactiveHistoryReplayBuffer),
                            ('noise', ReactiveHistoryNoiseReplayBuffer)]:
            for n_steps in (1, 3, 10):
                for history_length in (1, 4):
                    for bootstrap in (False, True):
                        kwargs = ({'base_action_dim': 4} if method == 'res' else {'exec_dim': 4})
                        buffer = cls(14,
                            gymnasium.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
                            gymnasium.spaces.Box(-1, 1, (4 if method == 'res' else 2,), dtype=np.float32),
                            device='cpu', n_envs=2, n_steps=n_steps, gamma=0.9,
                            handle_timeout_termination=bootstrap, action_step_dim=2,
                            history_length=history_length, **kwargs)
                        self.assertEqual(buffer.buffer_size, 7)
                        episodes = np.zeros(2, dtype=np.int64)
                        for t in range(11):
                            obs = np.array([[t, -t], [100 + t, -100 - t]], dtype=np.float32)
                            # Terminal snapshots differ visibly from reset observations.
                            done = np.array([t % 3 == 2, t % 4 == 3])
                            timeout = done & np.array([t % 2 == 0, t % 2 == 1])
                            next_obs = obs + np.where(done[:, None], 1000, 1)
                            action = np.arange(2 * buffer.action_dim, dtype=np.float32).reshape(2, -1) + t
                            executed = np.full((2, 4), t + 0.25, dtype=np.float32)
                            lengths = np.where(done, 1, 2)
                            executed[done, 2:] = 0
                            extra = dict(a_exec=executed, exec_steps=lengths, episode_id=episodes.copy())
                            if method == 'res':
                                extra.update(a_base=executed + 10, next_a_base=executed + 20)
                            buffer.add(obs, next_obs, action, np.array([t, 2 * t + 1], dtype=np.float32),
                                done, [{'TimeLimit.truncated': bool(value)} for value in timeout], **extra)
                            episodes += done
                            if t not in (4, 6, 10):
                                continue
                            indices = np.repeat(np.arange(buffer.buffer_size if buffer.full else buffer.pos), 2)
                            env_indices = np.tile([0, 1], len(indices) // 2)
                            expected = [scalar_sample(buffer, index, env_index)
                                        for index, env_index in zip(indices, env_indices)]
                            with patch('numpy.random.randint', side_effect=[indices, env_indices]):
                                batch = buffer.sample(len(indices))
                            for field, value in batch._asdict().items():
                                actual = value.cpu().numpy()
                                np.testing.assert_allclose(actual,
                                    np.stack([row[field] for row in expected]), rtol=1e-6, atol=1e-6,
                                    err_msg=str((method, n_steps, history_length, bootstrap, t, field)))
                                key = '{}_{}_{}_{}_{}_{}'.format(method, n_steps, history_length, int(bootstrap), t, field)
                                fixture[key] = actual
        # Write before/after snapshots only when explicitly requested by the caller.
        output = os.environ.get('ONELEG_REPLAY_FIXTURE_OUTPUT')
        if output:
            np.savez(output, **fixture)
        reference = os.environ.get('ONELEG_REPLAY_FIXTURE_REFERENCE')
        if reference:
            with np.load(reference) as previous:
                self.assertEqual(set(previous.files), set(fixture))
                for key, value in fixture.items():
                    self.assertEqual(value.dtype, previous[key].dtype)
                    if np.issubdtype(value.dtype, np.floating):
                        # NumPy versions can differ by one float32 ULP in power().
                        np.testing.assert_allclose(value, previous[key], rtol=2e-7, atol=1e-7, err_msg=key)
                    else:
                        np.testing.assert_array_equal(value, previous[key], err_msg=key)


if __name__ == '__main__':
    unittest.main()
