import unittest

import numpy as np
import torch

from tests.test_oneleg_online_history import make_replay


def episode(length, offset=0, timeout=True):
    result = []
    for step in range(length):
        value = float(offset + step)
        result.append(dict(obs=np.array([value], dtype=np.float32),
            next_obs=np.array([value + 1], dtype=np.float32),
            action=np.full(40, value, dtype=np.float32), reward=value,
            done=step == length - 1, timeout=timeout and step == length - 1,
            a_exec=np.full(40, value + .5, dtype=np.float32),
            exec_steps=np.int64(4), a_base=np.full(40, value + 10, dtype=np.float32),
            next_a_base=np.full(40, value + 11, dtype=np.float32)))
    return result


class OneLegEpisodeReplayTest(unittest.TestCase):
    def test_total_capacity_wrap_and_overwritten_history_never_reads_future(self):
        for method in ('resrl', 'dsrl'):
            rb = make_replay(method, capacity=7, n_step=3)
            for identity, length in enumerate((3, 4, 5, 2, 2)):
                items = episode(length, offset=identity * 10, timeout=identity % 2 == 0)
                if method == 'dsrl':
                    for item in items:
                        item['action'] = item['action'][:20]
                        del item['a_base'], item['next_a_base']
                rb.add_episode(items)
                self.assertEqual(rb.committed_episodes, identity + 1)
                storage = rb.storage
                self.assertEqual(storage.n_envs, 1)
                self.assertEqual(storage.buffer_size, 7)
                self.assertEqual(rb.size(), min(rb.committed_transitions, 7))
                ordered = ((storage.pos + np.arange(7)) % 7 if storage.full else np.arange(storage.pos))
                for position, index in enumerate(ordered):
                    batch = storage._get_samples(np.array([index]))
                    current_id = storage.episode_ids[index, 0]
                    past = []
                    for other in ordered[:position + 1][::-1]:
                        if storage.episode_ids[other, 0] != current_id:
                            break
                        past.append(other)
                    past = past[:5][::-1]
                    expected = torch.zeros(5, 1)
                    expected[-len(past):] = torch.as_tensor(storage.observations[past, 0])
                    torch.testing.assert_close(batch.obs[0], expected)
                    torch.testing.assert_close(batch.obs_valid[0], torch.tensor([False] * (5 - len(past)) + [True] * len(past)))
                    future = []
                    for other in ordered[position:position + 3]:
                        if storage.episode_ids[other, 0] != current_id:
                            break
                        future.append(other)
                        if storage.dones[other, 0]:
                            break
                    expected_reward = sum(.9 ** j * float(storage.rewards[other, 0]) for j, other in enumerate(future))
                    np.testing.assert_allclose(batch.rewards.item(), expected_reward, rtol=1e-6)
                    torch.testing.assert_close(batch.next_obs[0, -1], torch.as_tensor(storage.next_observations[future[-1], 0]))
                # Sampling remains uniform over stored transitions, not episode IDs.
                self.assertEqual(rb.sample(23).rewards.shape, (23, 1))

    def test_incomplete_or_oversized_episodes_are_rejected_before_writing(self):
        rb = make_replay('resrl', capacity=5)
        for items in (episode(6), episode(3)[:-1], []):
            with self.assertRaises(AssertionError):
                rb.add_episode(items)
            self.assertEqual(rb.size(), 0)
            self.assertEqual(rb.committed_transitions, 0)


if __name__ == '__main__':
    unittest.main()
