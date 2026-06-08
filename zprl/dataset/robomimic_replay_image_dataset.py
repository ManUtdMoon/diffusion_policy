if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from zprl.common.pytorch_util import dict_apply
from zprl.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from zprl.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from zprl.model.common.rotation_transformer import RotationTransformer
from zprl.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from zprl.common.replay_buffer import ReplayBuffer
from zprl.common.sampler import SequenceSampler, get_val_mask
from zprl.dataset.robomimic_image_util import (
    convert_robomimic_to_replay,
    create_image_sequence_sampler,
    create_train_val_mask,
    get_key_first_k,
    get_shape_meta_obs_keys,
)
from zprl.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()

class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            num_demo=None,
            n_prev_action_steps=0,
        ):
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        rgb_keys, lowdim_keys = get_shape_meta_obs_keys(shape_meta)
        obs_shapes = {
            key: tuple(attr['shape'])
            for key, attr in shape_meta['obs'].items()
        }

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + f'-n_{num_demo}' + '.zarr'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            dataset_path=dataset_path,
                            rgb_keys=rgb_keys,
                            lowdim_keys=lowdim_keys,
                            obs_shapes=obs_shapes,
                            action_shape=tuple(shape_meta['action']['shape']),
                            action_converter=lambda actions: _convert_actions(
                                raw_actions=actions,
                                abs_action=abs_action,
                                rotation_transformer=rotation_transformer,
                            ),
                            num_demo=num_demo,
                            img_compressor=None)
                        # convert to numpy backend for fast random access
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=replay_buffer.root.store, store=None)
                        print('Saving cache to disk.')
                        with zarr.DirectoryStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.DirectoryStore(cache_zarr_path) as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=None)  # numpy backend
                    print('Loaded!')
        else:
            replay_buffer = convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                dataset_path=dataset_path,
                rgb_keys=rgb_keys,
                lowdim_keys=lowdim_keys,
                obs_shapes=obs_shapes,
                action_shape=tuple(shape_meta['action']['shape']),
                action_converter=lambda actions: _convert_actions(
                    raw_actions=actions,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                ),
                num_demo=num_demo,
                img_compressor=None)
            # convert to numpy backend for fast random access
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=replay_buffer.root.store, store=None)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = get_key_first_k(n_obs_steps, rgb_keys, lowdim_keys)
        train_mask, _ = create_train_val_mask(replay_buffer, val_ratio, seed)
        sampler = create_image_sequence_sampler(
            replay_buffer=replay_buffer,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            train_mask=train_mask,
            key_first_k=key_first_k,
            prev_action_length=n_prev_action_steps)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self.n_prev_action_steps = n_prev_action_steps

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            prev_action_length=self.n_prev_action_steps
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]
        if self.n_prev_action_steps > 0:
            prev_action, prev_action_valid_mask = self.sampler.sample_prev_action(
                idx, action_start_offset=0 if self.n_obs_steps is None else self.n_obs_steps - 1)
            obs_dict['prev_action'] = prev_action.astype(np.float32)
            obs_dict['prev_action_valid_mask'] = prev_action_valid_mask.astype(np.float32)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data

    def __getitems__(self, indices: List[int]) -> List[Dict[str, torch.Tensor]]:
        """Batched fetch: vectorized obs reads for n_obs_steps=1."""
        threadpool_limits(1)

        if self.n_obs_steps != 1:
            return [self.__getitem__(idx) for idx in indices]

        indices_arr = np.asarray(indices, dtype=np.int64)
        batch_indices = self.sampler.indices[indices_arr]  # (B, 4)
        buffer_starts = batch_indices[:, 0]  # (B,)
        B = len(indices)

        # Batch read obs: arr[buffer_starts] is correct for k=1
        # regardless of padding (first obs frame is always arr[buffer_start])
        obs_batched = dict()
        for key in self.rgb_keys:
            arr = self.replay_buffer[key]
            frames = arr[buffer_starts]  # (B, H, W, C)
            frames = np.moveaxis(frames, -1, 1)  # (B, C, H, W)
            obs_batched[key] = frames[:, np.newaxis].astype(np.float32) / 255.  # (B,1,C,H,W)
        for key in self.lowdim_keys:
            arr = self.replay_buffer[key]
            obs_batched[key] = arr[buffer_starts][:, np.newaxis].astype(np.float32)  # (B,1,D)
        if self.n_prev_action_steps > 0:
            prev_actions = np.empty(
                (B, self.n_prev_action_steps) + self.replay_buffer['action'].shape[1:],
                dtype=np.float32)
            prev_action_valid_masks = np.empty((B, self.n_prev_action_steps), dtype=np.float32)
            for i, idx in enumerate(indices):
                prev_action, prev_action_valid_mask = self.sampler.sample_prev_action(
                    idx, action_start_offset=0 if self.n_obs_steps is None else self.n_obs_steps - 1)
                prev_actions[i] = prev_action.astype(np.float32)
                prev_action_valid_masks[i] = prev_action_valid_mask.astype(np.float32)
            obs_batched['prev_action'] = prev_actions
            obs_batched['prev_action_valid_mask'] = prev_action_valid_masks

        # Action: per-sample (small data, needs padding logic)
        # Only read 'action' key, not all keys
        action_arr = self.replay_buffer['action']
        actions = np.empty(
            (B, self.horizon) + action_arr.shape[1:], dtype=np.float32)
        for i, idx in enumerate(indices):
            index_row = self.sampler.indices[idx]
            data = self.sampler._sample_sequence_from_index_row(index_row, ['action'])
            actions[i] = data['action'].astype(np.float32)

        result = []
        for i in range(B):
            obs_dict = {key: torch.from_numpy(obs_batched[key][i]) for key in obs_batched}
            result.append({
                'obs': obs_dict,
                'action': torch.from_numpy(actions[i])
            })
        return result


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )



def main():
    task = "square"
    dataset_type = "mh"
    dataset_path = f"./data_local/robomimicv030/{task}/{dataset_type}/image_v141_subset_abs.hdf5"
    shape_meta = {
        "obs": {
            "agentview_image": {
                "shape": (3, 84, 84),
                "type": "rgb",
            },
            "robot0_eye_in_hand_image": {
                "shape": (3, 84, 84),
                "type": "rgb",
            },
            "robot0_eef_pos": {
                "shape": (3,),
            },
            "robot0_eef_quat": {
                "shape": (4,),
            },
            "robot0_gripper_qpos": {
                "shape": (2,),
            },
        },
        "action": {
            "shape": (10,),
        },
    }

    dataset = RobomimicReplayImageDataset(
        shape_meta,
        dataset_path=dataset_path,
        horizon=16,
        pad_before=0,
        pad_after=7,
        n_obs_steps=1,
        abs_action=True,
        rotation_rep='rotation_6d',
        use_cache=True,
        val_ratio=0.02,
        num_demo=100
    )

    val_set = dataset.get_validation_dataset()
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True,
        # pin_memory=True,
        # persistent_workers=False,
    )
    import time
    np.set_printoptions(precision=3)
    num_epochs = 1
    num_steps = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}:")
        train_time_per_batch = []
        start = time.time()
        for i, batch in enumerate(tqdm(train_loader)):
            time_get = time.time()
            train_time_per_batch.append(time_get - start)
            start = time_get
            if i + 1 == num_steps:
                break
        train = np.array(train_time_per_batch)
        print(f"Train mean: {train.mean():.3f}, std: {train.std():.3f}, max: {train.max():.3f}")
        print("train:", train[:10])

    # get action from replay buffer, median, mean
    normalizer = dataset.get_normalizer()
    action_normalizer = dataset.get_normalizer()['action']
    actions = np.array(dataset.replay_buffer['action'].astype(np.float32))
    print("NAction mean:", action_normalizer.normalize(actions.mean(axis=0)))
    print("NAction median:", action_normalizer.normalize(np.median(actions, axis=0)))

    epi_ends = dataset.replay_buffer.episode_ends[:]
    epi_lens = np.diff(np.concatenate([[0], epi_ends]))
    print("Episode length stats - mean:", epi_lens.mean(), "median:", np.median(epi_lens), "max:", epi_lens.max(), "min:", epi_lens.min(), "std:", epi_lens.std())

if __name__ == "__main__":
    main()
