"""Completed episodes in one transition stream; no training-environment axis."""

import numpy as np
import torch

from zprl.common.reactive_history_replay_buffer import ReactiveHistoryReplayBuffer
from zprl.common.reactive_history_noise_replay_buffer import ReactiveHistoryNoiseReplayBuffer


class OneLegEpisodeReplayBuffer:
    """Capacity is assumed larger than the longest episode, in decisions."""

    def __init__(self, *args, method='resrl', **kwargs):
        assert method in ('resrl', 'dsrl')
        self.storage = (ReactiveHistoryReplayBuffer if method == 'resrl' else ReactiveHistoryNoiseReplayBuffer)(
            *args, n_envs=1, **kwargs)
        self.committed_transitions = 0
        self.committed_episodes = 0

    def size(self):
        return self.storage.size()

    def sample(self, batch_size):
        return self.storage.sample(batch_size)

    def add_episode(self, episode):
        assert 0 < len(episode) <= self.storage.buffer_size
        assert episode[-1]['done'] and not any(item['done'] for item in episode[:-1])
        for item in episode:
            data = {key: np.expand_dims(value, 0) for key, value in item.items()
                    if key not in ('timeout', 'done')}
            self.storage.add(
                **data,
                done=np.array([item['done']]),
                infos=[{'TimeLimit.truncated': item['timeout']}],
                episode_id=np.array([self.committed_episodes], dtype=np.int64)
            )
        self.committed_transitions += len(episode)
        self.committed_episodes += 1


class OneLegEpisodeCollector:
    def __init__(self, replay: OneLegEpisodeReplayBuffer, n_envs: int):
        self.replay = replay
        self.n_envs = n_envs
        self.pending = [[] for _ in range(n_envs)]
        self.closed = np.ones(n_envs, dtype=bool)

    @property
    def pending_transitions(self):
        return sum(map(len, self.pending))

    def begin_batch(self):
        assert self.closed.all() and self.pending_transitions == 0
        self.closed[:] = False

    def collect(self, transition, rewards, dones, infos):
        def numpy(value):
            return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

        valid = numpy(transition['transition_valid']).astype(bool)
        assert valid.shape == (self.n_envs,)
        assert np.array_equal(valid, [info['transition_valid'] for info in infos])
        rows = np.flatnonzero(valid)
        data = {key: numpy(value[rows]) for key, value in transition.items()
                if key != 'transition_valid'}
        rewards, dones = numpy(rewards), numpy(dones)
        assert not np.any(dones & ~valid)
        for offset, row in enumerate(rows):
            assert not self.closed[row]
            item = {key: value[offset].copy() for key, value in data.items()}
            item.update(reward=rewards[row], done=bool(dones[row]),
                        timeout=bool(infos[row].get('TimeLimit.truncated', False)))
            self.pending[row].append(item)
            if dones[row]:
                self.replay.add_episode(self.pending[row])
                self.pending[row] = []
                self.closed[row] = True

    def discard_pending(self):
        self.pending = [[] for _ in range(self.n_envs)]
        self.closed[:] = True
