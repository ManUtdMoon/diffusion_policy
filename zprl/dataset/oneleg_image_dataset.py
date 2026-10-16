import copy
import os

import numpy as np
import torch
import zarr

from zprl.common.sampler import create_indices, get_val_mask
from zprl.common.replay_buffer import ReplayBuffer
from zprl.common.normalize_util import get_image_range_normalizer
from zprl.dataset.base_dataset import BaseImageDataset
from zprl.model.common.normalizer import LinearNormalizer


class OneLegImageDataset(BaseImageDataset):
    """OneLeg windows backed by read-only, compressed in-memory Zarr sources."""

    def __init__(self, shape_meta, zarr_paths, horizon, n_obs_steps=1,
            pad_before=0, pad_after=None, seed=42, val_ratio=0.05):
        if pad_after is None:
            pad_after = horizon - 1
        if n_obs_steps != 1 or horizon < 1:
            raise ValueError('OneLeg base requires n_obs_steps=1 and horizon >= 1')
        if pad_before != 0 or pad_after != horizon - 1:
            raise ValueError('OneLeg uses pad_before=0 and pad_after=horizon-1')
        if not 0 <= val_ratio < 1:
            raise ValueError('val_ratio must be in [0, 1)')
        expected_obs = {
            'color_image1': (3, 240, 320),
            'color_image2': (3, 240, 320),
            'robot_state': (16,),
        }
        if ({key: tuple(value['shape']) for key, value in shape_meta['obs'].items()}
                != expected_obs or tuple(shape_meta['action']['shape']) != (10,)):
            raise ValueError('OneLeg requires two CHW cameras, robot_state16 and action10')

        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.rgb_keys = ('color_image1', 'color_image2')
        self.zarr_paths = [os.path.expanduser(path) for path in zarr_paths]
        if not self.zarr_paths:
            raise ValueError('At least one Zarr source is required')
        self.sources = []
        for path in self.zarr_paths:
            buffer = ReplayBuffer.copy_from_path(path, store=zarr.MemoryStore(),
                keys=[*self.rgb_keys, 'robot_state', 'action'])
            self.sources.append(zarr.open_group(store=buffer.root.store, mode='r'))
        self.episode_ends = []
        self.train_masks = []
        self.val_masks = []
        for source in self.sources:
            ends = np.asarray(source['meta/episode_ends'][:], dtype=np.int64)
            if ends.ndim != 1 or not len(ends) or np.any(np.diff(np.r_[0, ends]) <= 0):
                raise ValueError('episode_ends must be strictly increasing and nonempty')
            for key, shape in expected_obs.items():
                array = source['data'][key]
                stored_shape = (240, 320, 3) if key in self.rgb_keys else shape
                if array.shape != (int(ends[-1]),) + stored_shape:
                    raise ValueError(f'Invalid stored shape for {key}: {array.shape}')
                expected_dtype = np.uint8 if key in self.rgb_keys else np.float32
                if array.dtype != expected_dtype:
                    raise ValueError(f'Invalid dtype for {key}: {array.dtype}')
            action = source['data/action']
            if action.shape != (int(ends[-1]), 10) or action.dtype != np.float32:
                raise ValueError('Stored action must be float32 (N, 10)')
            self.episode_ends.append(ends)
            val_mask = get_val_mask(len(ends), val_ratio, seed)
            self.val_masks.append(val_mask)
            self.train_masks.append(~val_mask)
        self.indices = self._make_indices(self.train_masks)
        weights = np.array([mask.sum() for mask in self.train_masks], dtype=np.float64)
        self.sample_probabilities = weights / weights.sum()
        self.validation = False

    def _make_indices(self, masks):
        return [create_indices(ends, self.horizon, mask,
                    pad_before=0, pad_after=self.horizon - 1).reshape(-1, 4)
                for ends, mask in zip(self.episode_ends, masks)]

    def get_validation_dataset(self, index=None):
        """Index selects one source, as in the author; None covers all sources."""
        dataset = copy.copy(self)
        dataset.indices = self._make_indices(self.val_masks)
        if index is not None:
            if not 0 <= index < len(self.sources):
                raise IndexError(index)
            dataset.indices = [rows if i == index else rows[:0]
                               for i, rows in enumerate(dataset.indices)]
        dataset.validation = True
        return dataset

    def get_normalizer(self):
        # The author fits global min/max across both full sources, including val.
        normalizer = LinearNormalizer()
        normalizer.fit({key: np.concatenate([source['data'][key][:]
                       for source in self.sources])
                       for key in ('action', 'robot_state')}, mode='limits')
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self):
        return torch.from_numpy(np.concatenate([
            source['data/action'][:] for source in self.sources]))

    def __len__(self):
        return sum(len(rows) for rows in self.indices)

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        if self.validation:
            source_index = 0
            while idx >= len(self.indices[source_index]):
                idx -= len(self.indices[source_index])
                source_index += 1
        else:
            source_index = int(np.random.choice(
                len(self.sources), p=self.sample_probabilities))
            idx %= len(self.indices[source_index])
        start, end, sample_start, sample_end = self.indices[source_index][idx]
        assert sample_start == 0
        data = self.sources[source_index]['data']
        obs = {key: torch.from_numpy(np.moveaxis(
            data[key][start:start + 1], -1, 1).astype(np.float32) / 255.)
            for key in self.rgb_keys}
        obs['robot_state'] = torch.from_numpy(data['robot_state'][start:start + 1])
        actions = np.empty((self.horizon, 10), dtype=np.float32)
        actions[:sample_end] = data['action'][start:end]
        actions[sample_end:] = actions[sample_end - 1]
        return {'obs': obs, 'action': torch.from_numpy(actions)}
