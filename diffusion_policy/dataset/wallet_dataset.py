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
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    array_to_stats
)
register_codecs()

class WalletDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            num_demo=None,
            # Optional per-sample RGB augmentation applied in __getitem__.
            # Callable taking a [T, C, H, W] float tensor in [0, 1] and
            # returning the same shape. One set of params per call, so
            # augmentation is time-consistent across the T frames of a
            # single sample. Automatically disabled on the validation set.
            rgb_aug=None,
        ):
        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + f'-n_{num_demo}' + '.zarr'
            print(f'Using cache at {cache_zarr_path}')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_h5_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            num_demo=num_demo)
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
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_h5_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                num_demo=num_demo)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.rgb_aug = rgb_aug

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
        val_set.rgb_aug = None
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        normalizer['action'] = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            this_normalizer = get_range_normalizer_from_stat(stat)
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
            # T,H,W,C(rgb) -> T,C,H,W, float32 in [0, 1]
            imgs = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            imgs = torch.from_numpy(imgs)
            if self.rgb_aug is not None:
                # one set of aug params per call -> time-consistent across T
                imgs = self.rgb_aug(imgs)
            obs_dict[key] = imgs
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = torch.from_numpy(data[key][T_slice].astype(np.float32))
            del data[key]

        torch_data = {
            'obs': obs_dict,
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data


def _convert_h5_to_replay(
        store,
        shape_meta,
        dataset_path, 
        n_workers=None,
        max_inflight_tasks=None,
        num_demo=None
    ):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file
        episode_ends = list()
        prev_end = 0
        n_demo = len(demos) if num_demo is None else min(num_demo, len(demos))
        print("Total number of demos:", n_demo)
        for i in range(n_demo):
            demo = demos[f'demo_{i}']
            episode_length = demo['action'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = key
            if key == 'action':
                data_key = 'action_quat'
            this_data = list()
            for i in range(n_demo):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = f"images/{key}"
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(n_demo):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo[data_key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def main():
    task = 'wallet'
    dataset_path = f"/media/datahub-2/ydj/real/wallet/{task}.h5"
    shape_meta = {
        "obs": {
            "global": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "wrist_0": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "wrist_1": {
                "shape": (3, 120, 160),
                "type": "rgb",
            },
            "qpos": {
                "shape": (16,),  # 7 + 1 + 7 + 1
                "type": "low_dim",
            },
        },
        "action": {
            "shape": (16,), # 2 * (3 + 4 + 1)
        },
    }

    dataset = WalletDataset(
        shape_meta,
        dataset_path=dataset_path,
        horizon=15,
        pad_before=0,
        pad_after=14,
        n_obs_steps=1,
        use_cache=True,
        val_ratio=0.02,
        num_demo=50
    )

    val_set = dataset.get_validation_dataset()
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=0,
        shuffle=True,
        # pin_memory=True,
        # persistent_workers=False,
    )
    import time
    np.set_printoptions(precision=3)
    num_epochs = 1
    num_steps = 5
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
    print("NAction min:", action_normalizer.normalize(actions.min(axis=0)))
    print("NAction max:", action_normalizer.normalize(actions.max(axis=0)))

    epi_ends = dataset.replay_buffer.episode_ends[:]
    epi_lens = np.diff(np.concatenate([[0], epi_ends]))
    print("Episode length stats - mean:", epi_lens.mean(), "median:", np.median(epi_lens), "max:", epi_lens.max(), "min:", epi_lens.min(), "std:", epi_lens.std())

if __name__ == "__main__":
    main()
