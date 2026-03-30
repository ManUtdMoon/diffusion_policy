from typing import Dict, List, Optional
import concurrent.futures
import multiprocessing

import h5py
import numpy as np
from tqdm import tqdm
import zarr

from zprl.codecs.imagecodecs_numcodecs import Jpeg2k
from zprl.common.replay_buffer import ReplayBuffer
from zprl.common.sampler import SequenceSampler, get_val_mask


def get_shape_meta_obs_keys(shape_meta: dict):
    rgb_keys = list()
    lowdim_keys = list()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    return rgb_keys, lowdim_keys


def get_key_first_k(n_obs_steps: Optional[int], rgb_keys: List[str], lowdim_keys: List[str]):
    key_first_k = dict()
    if n_obs_steps is not None:
        for key in rgb_keys + lowdim_keys:
            key_first_k[key] = n_obs_steps
    return key_first_k


def create_image_sequence_sampler(
        replay_buffer: ReplayBuffer,
        horizon: int,
        pad_before: int,
        pad_after: int,
        train_mask: np.ndarray,
        key_first_k: Dict[str, int]):
    return SequenceSampler(
        replay_buffer=replay_buffer,
        sequence_length=horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        episode_mask=train_mask,
        key_first_k=key_first_k)


def create_train_val_mask(replay_buffer: ReplayBuffer, val_ratio: float, seed: int):
    val_mask = get_val_mask(
        n_episodes=replay_buffer.n_episodes,
        val_ratio=val_ratio,
        seed=seed)
    train_mask = ~val_mask
    return train_mask, val_mask


def convert_robomimic_to_replay(
        store,
        dataset_path: str,
        rgb_keys: List[str],
        lowdim_keys: List[str],
        obs_shapes: Dict[str, tuple],
        action_shape: tuple,
        action_converter,
        n_workers=None,
        max_inflight_tasks=None,
        num_demo=None,
        img_compressor='default'):
    if img_compressor == 'default':
        img_compressor = Jpeg2k(level=50)
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        n_demo = len(demos) if num_demo is None else min(num_demo, len(demos))
        print("Total number of demos:", n_demo)
        for i in range(n_demo):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array(
            'episode_ends',
            episode_ends,
            dtype=np.int64,
            compressor=None,
            overwrite=True)

        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(n_demo):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = action_converter(this_data)
                assert this_data.shape == (n_steps,) + tuple(action_shape)
            else:
                assert this_data.shape == (n_steps,) + tuple(obs_shapes[key])
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
                _ = zarr_arr[zarr_idx]
                return True
            except Exception:
                return False

        with tqdm(total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    shape = tuple(obs_shapes[key])
                    c, h, w = shape
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=img_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(n_demo):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(executor.submit(
                                img_copy,
                                img_arr,
                                zarr_idx,
                                hdf5_arr,
                                hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    return ReplayBuffer(root)
