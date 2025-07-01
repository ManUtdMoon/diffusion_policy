import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import dill
import pickle
from datetime import datetime
from copy import deepcopy
import numpy as np
import torch
import zarr
import hydra
from tqdm import tqdm
import shutil

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.policy.flow_match_unet_image_policy import FlowMatchUnetImagePolicy
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from rnd.model import RND

register_codecs()

def extract_obs_dict(replay_buffer, obs_keys, episode_idx, step_idx, n_obs_steps):
    """
    Extract observation dictionary for the last n_obs_steps observations.
    
    Args:
        replay_buffer: ReplayBuffer containing the trajectory data
        obs_keys: List of observation keys to extract
        episode_idx: Episode index
        step_idx: Current step index within the episode
        n_obs_steps: Number of observation steps to collect
    
    Returns:
        obs_dict: Dictionary containing observations in robomimic format
    """
    # Get episode boundaries
    episode_ends = replay_buffer.episode_ends[:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    
    episode_start = episode_starts[episode_idx]
    episode_end = episode_ends[episode_idx]
    
    # Calculate the absolute To index in the dataset
    abs_step_idx = episode_start + step_idx
    
    # Calculate observation window index 
    T_sub = len(step_idx)
    To = n_obs_steps
    abs_obs_idxs = np.tile(abs_step_idx, (To, 1)).T # (T_sub, To)
    abs_obs_idxs -= np.arange(To - 1, -1, -1)
    abs_obs_idxs = np.clip(abs_obs_idxs, episode_start, episode_end - 1)
    abs_obs_idxs = abs_obs_idxs.flatten()
    
    obs_dict = {}
    for key in obs_keys:
        data = replay_buffer[key][abs_obs_idxs].astype(np.float32) # (T_sub*To, ...)
        if data.ndim > 2: # images
            data = np.moveaxis(data, -1, 1) / 255.0
        obs_dict[key] = data.reshape(T_sub, To, *data.shape[1:])  # (T_sub, To, ...)

    return dict_apply(obs_dict, torch.from_numpy)


def _get_band_modulation(mean_trajs, band_mean, CI=0.95, mod_type='const'):
    assert 0 <= CI <= 1, f"CI must be in [0, 1], got {CI}"
    assert mod_type in ['const', 'tfunc'], \
        f"mod_type must be one of ['const', 'tfunc'], got {mod_type}"
    eps = 1e-8
    length = mean_trajs.shape[-1] # T_sub

    if mod_type == 'const':
        band_mod = np.ones((1, length), dtype=np.float32) / length  # (1,T_sub)
    elif mod_type == 'tfunc':
        N1 = mean_trajs.shape[0]
        if (int(np.ceil((N1 + 1) * CI)) > N1):
            band_mod = np.max(
                np.abs(mean_trajs - band_mean), axis=0, keepdims=True
            ) + eps
        else:
            dev = np.abs(mean_trajs - band_mean) # (N1,T_sub)
            gamma = np.quantile(dev.max(axis=1), CI)  # ()
            inlier_mask = dev.max(axis=1) <= gamma # (N1,)
            band_mod = dev[inlier_mask].max(axis=0, keepdims=True) + eps # (1,T_sub)

    return band_mod


@click.command()
@click.option('--rnd_ckpt', '-c', required=True,
    help='Path to trained RND checkpoint')
@click.option('--mean_ratio', '-r', default=0.3, type=float,
    help='Ratio of trajs for mean')
@click.option('--n_action_steps', '-Ta', default=4, type=int, 
    help='Action horizon (number of action steps to execute)')
@click.option('--device', '-d', default='cuda:0')
@click.option('--mod_type', '-m', default='const', type=str)
@click.option('--confidence_interval', '-ci', default=0.95, type=float)
def main(rnd_ckpt, mean_ratio, n_action_steps, device, mod_type, confidence_interval):
    """
    Loop over trajectories in zarr dataset and collect observations at regular intervals.
    
    This script processes trajectories from a zarr dataset, extracting observation
    dictionaries at every n_action_steps interval, with each obs_dict containing
    the last n_obs_steps observations in robomimic format.
    """
    # 1. load stuff
    device = torch.device(device)
    output_dir = str(pathlib.Path(rnd_ckpt).parent / 'calib')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1.1 load rnd
    print(f"Loading RND checkpoint from {rnd_ckpt}...")
    rnd_payload = torch.load(open(rnd_ckpt, 'rb'), pickle_module=dill)
    rnd_cfg = deepcopy(rnd_payload['config'])
    rnd = RND(
        input_dim=rnd_cfg['input_dim'],
        hidden_dims=rnd_cfg['hidden_dims'],
        output_dim=rnd_cfg['output_dim'],
    )
    rnd.load_state_dict(rnd_payload['model'])
    rnd.to(device)
    rnd.eval()
    rnd.requires_grad_(False)

    # 1.2 load base_policy
    policy_ckpt = rnd_cfg['policy']
    print(f"Loading base policy from {policy_ckpt}...")
    base_payload = torch.load(open(policy_ckpt, 'rb'), pickle_module=dill)
    base_cfg = base_payload['cfg']
    base: FlowMatchUnetImagePolicy = hydra.utils.instantiate(base_cfg.policy)
    base.load_state_dict(base_payload['state_dicts']['ema_model'])
    base.to(device)
    base.eval()
    base.requires_grad_(False)

    # 1.3 load dataset into replay buffer
    task = base_cfg.task_name
    dataset_path = str(pathlib.Path(f'data/rnd/{task}_100_calib.zarr.zip'))
    print(f"Loading dataset from {dataset_path}...")
    with zarr.ZipStore(dataset_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, store=zarr.MemoryStore())
    print("Lengths percentile:", np.percentile(replay_buffer.episode_lengths, [0, 25, 50, 75, 100]))

    # 1.4 computing necessary constants
    ## numbers
    Ta = n_action_steps
    To = base_cfg.n_obs_steps
    T = base_cfg.task.env_runner.max_steps
    n_trajs = replay_buffer.n_episodes
    n_mean_trajs = int(mean_ratio * n_trajs)

    ## shapes and arrays
    obs_shape_meta = base_cfg.shape_meta.obs
    obs_keys = list(obs_shape_meta.keys())
    step_idx = np.arange(0, T, Ta)
    T_sub = len(step_idx)  # Number of steps sampled from each trajectory

    # 2. process trajectories
    cache_score_path = pathlib.Path(output_dir) / f'{task}_rnd_score.npy'
    if not pathlib.Path(cache_score_path).exists():
        try:
            print('Cache does not exist. Creating!')
            scores = np.full((n_trajs, T_sub), np.nan, dtype=np.float32)  # (N, T_sub)
            for i in tqdm(range(n_trajs), desc="Processing trajectories"):
                # each value: (T_sub, To, ...)
                traj_obs_dict = extract_obs_dict(
                    replay_buffer, episode_idx=i, obs_keys=obs_keys, step_idx=step_idx, n_obs_steps=To)

                # forward through base for obs_emb
                traj_obs_dict = dict_apply(
                    traj_obs_dict, lambda x: x.to(device, non_blocking=True))
                traj_obs_embs = base.encode_obs(traj_obs_dict) # (T_sub, Do)

                # forward through rnd for score
                traj_scores = rnd(traj_obs_embs).cpu().numpy() # (T_sub,)
                scores[i] = traj_scores
            np.save(cache_score_path, scores)
        except Exception as e:
            shutil.rmtree(cache_score_path)
            raise e
    else:
        print(f"Loading cached scores from {cache_score_path}...")
        scores = np.load(cache_score_path)

    with np.printoptions(precision=2, suppress=True):
        print(scores)

    # 3. calibration for band mean and upper bound (mean + std)
    ## 3.1 select trajectory indices
    CI = confidence_interval
    mean_traj_idxs = np.arange(n_mean_trajs)
    std_traj_idxs = np.arange(n_mean_trajs, n_trajs)
    mean_trajs = scores[mean_traj_idxs]  # (n_mean, T_sub)
    std_trajs = scores[std_traj_idxs]  # (N-n_mean, T_sub)

    band_mean = np.mean(mean_trajs, axis=0, keepdims=True)  # (1,T_sub)
    band_modulation = _get_band_modulation(
        mean_trajs, band_mean, CI=CI, mod_type=mod_type)  # (1,T_sub), s_cal in paper
    deviation = np.max((std_trajs - band_mean) / band_modulation, axis=1) # (N2,)
    band_width = np.quantile(deviation, CI)  # ()
    bound = band_mean + band_width * band_modulation  # (1,T_sub)

    with np.printoptions(precision=3, suppress=True):
        print(f"Band mean: {band_mean.squeeze()}")
        print(f"Band modulation: {band_modulation.squeeze()}")
        print(f"Band width: {band_width}")
        print(f"Bound: {bound.squeeze()}")

    # 4. save results
    calib_data = {
        'band_mean': band_mean.squeeze(),
        'band_modulation': band_modulation.squeeze(),
        'band_width': band_width,
        'bound': bound.squeeze(),
        'CI': CI,
    }
    calib_path = pathlib.Path(output_dir) / f'{task}_calib_band_{CI}_{mod_type}.pkl'
    with open(calib_path, 'wb') as f:
        pickle.dump(calib_data, f)


if __name__ == "__main__":
    main()
