from typing import Dict
import os
import copy

import numpy as np
import torch

from zprl.common.pytorch_util import dict_apply
from zprl.common.replay_buffer import ReplayBuffer
from zprl.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from zprl.model.common.normalizer import LinearNormalizer
from zprl.dataset.base_dataset import BaseImageDataset
from zprl.common.normalize_util import (
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)


class MetaworldImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,
            horizon=16,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        if not os.path.isdir(zarr_path):
            raise FileNotFoundError(f"MetaWorld dataset not found: {zarr_path}")

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'img'])

        # ReplayBuffer.copy_from_path uses an in-memory numpy backend here.
        # Clip the copied expert actions without modifying the source zarr.
        np.clip(self.replay_buffer['action'], -1.0, 1.0,
            out=self.replay_buffer['action'])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()

        stat = array_to_stats(self.replay_buffer['state'])
        normalizer['agent_pos'] = get_range_normalizer_from_stat(stat)

        stat = array_to_stats(self.replay_buffer['action'])
        normalizer['action'] = get_identity_normalizer_from_stat(stat)

        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        T_slice = slice(self.n_obs_steps)
        agent_pos = sample['state'][T_slice].astype(np.float32)
        image = sample['img'][T_slice].astype(np.float32) / 255.0

        data = {
            'obs': {
                'image': np.moveaxis(image, -1, 1),
                'agent_pos': agent_pos,
            },
            'action': sample['action'].astype(np.float32)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
