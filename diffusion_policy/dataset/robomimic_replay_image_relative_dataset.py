if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


from typing import Dict
import copy
import hashlib
import json
import os
import shutil

import numpy as np
import torch
import zarr
from filelock import FileLock
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pose_util import mat_to_pose10d
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.robomimic_image_util import (
    convert_robomimic_to_replay,
    create_image_sequence_sampler,
    create_train_val_mask,
    get_key_first_k,
    get_shape_meta_obs_keys,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

register_codecs()


def _to_jsonable(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _resolve_raw_obs_key(key: str) -> str:
    if key.endswith('_rot6d'):
        return key[:-len('_rot6d')] + '_quat'
    return key


def _build_pose_mat(pos: np.ndarray, rot_mat: np.ndarray) -> np.ndarray:
    mats = np.zeros(pos.shape[:-1] + (4, 4), dtype=np.float32)
    mats[..., :3, :3] = rot_mat.astype(np.float32)
    mats[..., :3, 3] = pos.astype(np.float32)
    mats[..., 3, 3] = 1.0
    return mats


class RobomimicReplayImageRelativeDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            use_cache=False,
            sample_cache_build_batch_size=128,
            seed=42,
            val_ratio=0.0,
            num_demo=None,
        ):
        rgb_keys, lowdim_keys = get_shape_meta_obs_keys(shape_meta)
        raw_lowdim_keys = list(dict.fromkeys(
            _resolve_raw_obs_key(key) for key in lowdim_keys
        ))
        obs_shapes = {
            key: tuple(attr['shape'])
            for key, attr in shape_meta['obs'].items()
            if attr.get('type', 'low_dim') == 'rgb'
        }
        for key in lowdim_keys:
            raw_key = _resolve_raw_obs_key(key)
            if key.endswith('_rot6d'):
                obs_shapes[raw_key] = (4,)
            else:
                obs_shapes[raw_key] = tuple(shape_meta['obs'][key]['shape'])

        replay_buffer = None
        if use_cache:
            cache_spec = {
                'path': os.path.expanduser(dataset_path),
                'num_demo': num_demo,
                'raw_lowdim_keys': raw_lowdim_keys,
            }
            cache_hash = hashlib.md5(
                json.dumps(cache_spec, sort_keys=True).encode('utf-8')
            ).hexdigest()[:8]
            cache_zarr_path = dataset_path + f'-n_{num_demo}-relative-{cache_hash}.zarr'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            dataset_path=dataset_path,
                            rgb_keys=rgb_keys,
                            lowdim_keys=raw_lowdim_keys,
                            obs_shapes=obs_shapes,
                            action_shape=(7,),
                            action_converter=lambda actions: actions.astype(np.float32),
                            num_demo=num_demo)
                        print('Saving cache to disk.')
                        with zarr.DirectoryStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.DirectoryStore(cache_zarr_path) as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                dataset_path=dataset_path,
                rgb_keys=rgb_keys,
                lowdim_keys=raw_lowdim_keys,
                obs_shapes=obs_shapes,
                action_shape=(7,),
                action_converter=lambda actions: actions.astype(np.float32),
                num_demo=num_demo)

        key_first_k = get_key_first_k(n_obs_steps, rgb_keys, raw_lowdim_keys)
        train_mask, _ = create_train_val_mask(replay_buffer, val_ratio, seed)
        sampler = create_image_sequence_sampler(
            replay_buffer=replay_buffer,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            train_mask=train_mask,
            key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.raw_lowdim_keys = raw_lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.obs_horizon = n_obs_steps if n_obs_steps is not None else horizon
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_cache = use_cache
        self.sample_cache_build_batch_size = sample_cache_build_batch_size
        self.dataset_path = os.path.expanduser(dataset_path)
        self.num_demo = num_demo
        self.seed = seed
        self.val_ratio = val_ratio
        self.rotation_transformer = RotationTransformer('quaternion', 'matrix')
        self.raw_action_rotation_transformer = RotationTransformer('axis_angle', 'matrix')

        self.pose_robot_prefixes = list()
        for key in self.lowdim_keys:
            if key.endswith('_eef_pos'):
                prefix = key[:-len('_eef_pos')]
                if f'{prefix}_eef_rot6d' in self.shape_meta['obs']:
                    self.pose_robot_prefixes.append(prefix)
        if len(self.pose_robot_prefixes) != 1:
            raise RuntimeError('relative robomimic image dataset currently supports single-arm only')

        self.sample_cache_group = None
        self.sample_cache_path = None
        self.rgb_sampler = None
        self._refresh_runtime_views()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        key_first_k = get_key_first_k(self.n_obs_steps, self.rgb_keys, self.raw_lowdim_keys)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            key_first_k=key_first_k)
        val_set.train_mask = ~self.train_mask
        val_set._refresh_runtime_views()
        return val_set

    def _refresh_runtime_views(self):
        self.sample_cache_group = None
        self.rgb_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.train_mask,
            keys=self.rgb_keys,
            key_first_k=get_key_first_k(self.n_obs_steps, self.rgb_keys, []))
        if self.use_cache:
            self.sample_cache_path = self._get_sample_cache_path()
        else:
            self.sample_cache_path = None

    def _get_sample_cache_path(self):
        cache_spec = {
            'path': self.dataset_path,
            'num_demo': self.num_demo,
            'shape_meta': _to_jsonable(self.shape_meta),
            'lowdim_keys': self.lowdim_keys,
            'n_obs_steps': self.n_obs_steps,
            'horizon': self.horizon,
            'pad_before': self.pad_before,
            'pad_after': self.pad_after,
            'indices_hash': hashlib.md5(self.sampler.indices.tobytes()).hexdigest(),
        }
        cache_hash = hashlib.md5(
            json.dumps(cache_spec, sort_keys=True).encode('utf-8')
        ).hexdigest()[:10]
        return self.dataset_path + f'-n_{self.num_demo}-relative-samples-{cache_hash}.zarr'

    def _ensure_sample_cache(self):
        if self.sample_cache_path is None:
            return None
        if self.sample_cache_group is not None:
            return self.sample_cache_group

        cache_lock_path = self.sample_cache_path + '.lock'
        with FileLock(cache_lock_path):
            if not os.path.exists(self.sample_cache_path):
                print('Building transformed sample cache.')
                self._build_sample_cache()
        self.sample_cache_group = zarr.open_group(self.sample_cache_path, mode='r')
        return self.sample_cache_group

    def _build_sample_cache(self):
        obs_shapes = {
            key: (len(self.sampler), self.obs_horizon) + tuple(self.shape_meta['obs'][key]['shape'])
            for key in self.lowdim_keys
        }
        action_shape = (len(self.sampler), self.horizon) + tuple(self.shape_meta['action']['shape'])

        root = zarr.open_group(self.sample_cache_path, mode='w')
        meta_group = root.require_group('meta', overwrite=True)
        data_group = root.require_group('data', overwrite=True)
        _ = meta_group.array(
            'indices',
            self.sampler.indices.astype(np.int64),
            dtype=np.int64,
            compressor=None,
            overwrite=True)

        arrays = dict()
        for key, shape in obs_shapes.items():
            arrays[key] = data_group.zeros(
                name=key,
                shape=shape,
                chunks=(1,) + shape[1:],
                dtype=np.float32,
                compressor=None,
                overwrite=True)
        arrays['action'] = data_group.zeros(
            name='action',
            shape=action_shape,
            chunks=(1,) + action_shape[1:],
            dtype=np.float32,
            compressor=None,
            overwrite=True)

        batch_size = max(1, int(self.sample_cache_build_batch_size))
        n_samples = len(self.sampler)
        for batch_start in tqdm(
                range(0, n_samples, batch_size),
                desc='building transformed sample cache'):
            batch_end = min(batch_start + batch_size, n_samples)
            data_batch = self.sampler.sample_sequence_batch(
                indices=np.arange(batch_start, batch_end, dtype=np.int64),
                keys=self.raw_lowdim_keys + ['action'])
            transformed = self._transform_sample_batch(
                data_batch,
                include_rgb=False)
            for key in self.lowdim_keys:
                arrays[key][batch_start:batch_end] = transformed['obs'][key]
            arrays['action'][batch_start:batch_end] = transformed['action']

    def _get_cached_sample(self, idx: int):
        cache_group = self._ensure_sample_cache()
        if cache_group is None:
            return None
        data_group = cache_group['data']
        obs_dict = {
            key: data_group[key][idx].astype(np.float32)
            for key in self.lowdim_keys
        }
        action = data_group['action'][idx].astype(np.float32)
        return {
            'obs': obs_dict,
            'action': action,
        }

    def _transform_sample(self, data: Dict[str, np.ndarray], include_rgb: bool) -> Dict[str, np.ndarray]:
        batch_data = {
            key: value[None]
            for key, value in data.items()
        }
        transformed = self._transform_sample_batch(batch_data, include_rgb=include_rgb)
        return {
            'obs': {key: value[0] for key, value in transformed['obs'].items()},
            'action': transformed['action'][0],
        }

    def _transform_sample_batch(self, data: Dict[str, np.ndarray], include_rgb: bool) -> Dict[str, np.ndarray]:
        T_slice = slice(self.obs_horizon)

        obs_dict = dict()
        if include_rgb:
            for key in self.rgb_keys:
                obs_dict[key] = np.moveaxis(data[key][:, T_slice], -1, 2).astype(np.float32) / 255.

        raw_obs = {
            key: data[key][:, T_slice].astype(np.float32)
            for key in self.raw_lowdim_keys
        }

        anchor_idx = self.obs_horizon - 1
        raw_action = data['action'].astype(np.float32)
        anchor_pose_mat_dict = dict()
        for prefix in self.pose_robot_prefixes:
            pos_key = f'{prefix}_eef_pos'
            quat_key = f'{prefix}_eef_quat'
            rot_key = f'{prefix}_eef_rot6d'

            rot_mat = self.rotation_transformer.forward(raw_obs[quat_key])
            pose_mat = _build_pose_mat(raw_obs[pos_key], rot_mat)
            anchor_pose_mat = pose_mat[:, anchor_idx].copy()
            anchor_pose_mat_dict[prefix] = anchor_pose_mat
            rel_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=anchor_pose_mat[:, None],
                pose_rep='relative',
                backward=False)
            rel_pose = mat_to_pose10d(rel_pose_mat)
            obs_dict[pos_key] = rel_pose[..., :3].astype(np.float32)
            obs_dict[rot_key] = rel_pose[..., 3:].astype(np.float32)

        action_pos = raw_action[..., :3]
        action_rot = raw_action[..., 3:6]
        action_gripper = raw_action[..., 6:]
        action_rot_mat = self.raw_action_rotation_transformer.forward(action_rot)
        action_pose_mat = _build_pose_mat(action_pos, action_rot_mat)
        anchor_prefix = self.pose_robot_prefixes[0]
        anchor_pose_mat = anchor_pose_mat_dict[anchor_prefix]
        rel_action_mat = convert_pose_mat_rep(
            action_pose_mat,
            base_pose_mat=anchor_pose_mat[:, None],
            pose_rep='relative',
            backward=False)
        rel_action = mat_to_pose10d(rel_action_mat).astype(np.float32)
        transformed_action = np.concatenate([rel_action, action_gripper], axis=-1).astype(np.float32)

        for key in self.lowdim_keys:
            if key in obs_dict:
                continue
            raw_key = _resolve_raw_obs_key(key)
            obs_dict[key] = raw_obs[raw_key].astype(np.float32)

        return {
            'obs': obs_dict,
            'action': transformed_action,
        }

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        if self.use_cache:
            cache_group = self._ensure_sample_cache()
            data_cache = {
                key: cache_group['data'][key][:].reshape(-1, self.shape_meta['obs'][key]['shape'][0])
                for key in self.lowdim_keys
            }
            data_cache['action'] = cache_group['data']['action'][:].reshape(
                -1, self.shape_meta['action']['shape'][0])
        else:
            data_cache = {key: list() for key in self.lowdim_keys + ['action']}
            threadpool_limits(1)
            for idx in tqdm(range(len(self.sampler)), desc='iterating dataset to get normalization'):
                data = self.sampler.sample_sequence(idx)
                transformed = self._transform_sample(data, include_rgb=False)
                for key in self.lowdim_keys:
                    data_cache[key].append(transformed['obs'][key])
                data_cache['action'].append(transformed['action'])

            for key, values in data_cache.items():
                data_cache[key] = np.concatenate(values, axis=0)

        action_stat = array_to_stats(data_cache['action'])
        normalizer['action'] = _make_relative_action_normalizer(action_stat)

        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])
            if key.endswith('_eef_pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('_eef_rot6d'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('_gripper_qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f'unsupported lowdim key {key}')
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        cached = self._get_cached_sample(0) if self.use_cache and len(self.sampler) > 0 else None
        if cached is not None:
            cache_group = self._ensure_sample_cache()
            actions = cache_group['data']['action'][:].reshape(-1, self.shape_meta['action']['shape'][0])
            return torch.from_numpy(actions.astype(np.float32))

        all_actions = list()
        for idx in range(len(self.sampler)):
            data = self.sampler.sample_sequence(idx)
            transformed = self._transform_sample(data, include_rgb=False)
            all_actions.append(transformed['action'])
        return torch.from_numpy(np.concatenate(all_actions, axis=0))

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        transformed = self._get_cached_sample(idx)
        if transformed is None:
            data = self.sampler.sample_sequence(idx)
            transformed = self._transform_sample(data, include_rgb=False)
        rgb_data = self.rgb_sampler.sample_sequence(idx)
        for key in self.rgb_keys:
            transformed['obs'][key] = np.moveaxis(
                rgb_data[key][:self.obs_horizon], -1, 1
            ).astype(np.float32) / 255.
        return {
            'obs': dict_apply(transformed['obs'], torch.from_numpy),
            'action': torch.from_numpy(transformed['action'].astype(np.float32))
        }


def _make_relative_action_normalizer(stat):
    pos_norm = get_range_normalizer_from_stat({
        name: value[..., :3] for name, value in stat.items()
    })
    rot_norm = get_identity_normalizer_from_stat({
        name: value[..., 3:9] for name, value in stat.items()
    })
    gripper_norm = get_range_normalizer_from_stat({
        name: value[..., 9:] for name, value in stat.items()
    })

    scale = np.concatenate([
        pos_norm.params_dict['scale'].detach().cpu().numpy(),
        rot_norm.params_dict['scale'].detach().cpu().numpy(),
        gripper_norm.params_dict['scale'].detach().cpu().numpy(),
    ], axis=0)
    offset = np.concatenate([
        pos_norm.params_dict['offset'].detach().cpu().numpy(),
        rot_norm.params_dict['offset'].detach().cpu().numpy(),
        gripper_norm.params_dict['offset'].detach().cpu().numpy(),
    ], axis=0)
    input_stats = {
        name: np.concatenate([
            pos_norm.params_dict['input_stats'][name].detach().cpu().numpy(),
            rot_norm.params_dict['input_stats'][name].detach().cpu().numpy(),
            gripper_norm.params_dict['input_stats'][name].detach().cpu().numpy(),
        ], axis=0)
        for name in ['min', 'max', 'mean', 'std']
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=input_stats
    )


def main():
    task = "square"
    dataset_type = "mh"
    dataset_path = f"/media/datahub-2/ydj/robomimicv030/{task}/{dataset_type}/image_v141_subset_abs.hdf5"
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
            "robot0_eef_rot6d": {
                "shape": (6,),
            },
            "robot0_gripper_qpos": {
                "shape": (2,),
            },
        },
        "action": {
            "shape": (10,),
        },
    }

    dataset = RobomimicReplayImageRelativeDataset(
        shape_meta,
        dataset_path=dataset_path,
        horizon=16,
        pad_before=0,
        pad_after=7,
        n_obs_steps=1,
        use_cache=True,
        val_ratio=0.02,
        num_demo=100
    )

    _ = dataset.get_validation_dataset()
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True,
    )

    import time
    np.set_printoptions(precision=3, suppress=True)
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
            print("obs keys:", list(batch['obs'].keys()))
            print("action shape:", tuple(batch['action'].shape))
            for key, value in batch['obs'].items():
                print(f"{key} shape:", tuple(value.shape))
            if i + 1 == num_steps:
                break
        train = np.array(train_time_per_batch)
        print(f"Train mean: {train.mean():.3f}, std: {train.std():.3f}, max: {train.max():.3f}")
        print("train:", train[:10])

    normalizer = dataset.get_normalizer()
    action_normalizer = normalizer['action']
    actions = dataset.get_all_actions().numpy().astype(np.float32)
    print("Action mean:", actions.mean(axis=0))
    print("NAction mean:", action_normalizer.normalize(actions.mean(axis=0)))
    print("NAction median:", action_normalizer.normalize(np.median(actions, axis=0)))

    epi_ends = dataset.replay_buffer.episode_ends[:]
    epi_lens = np.diff(np.concatenate([[0], epi_ends]))
    print(
        "Episode length stats - mean:",
        epi_lens.mean(),
        "median:",
        np.median(epi_lens),
        "max:",
        epi_lens.max(),
        "min:",
        epi_lens.min(),
        "std:",
        epi_lens.std()
    )


if __name__ == "__main__":
    main()
