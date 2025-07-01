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
from diffusion_policy.env_runner.robomimic_image_runner_with_detector import RobomimicImageRunnerWithDetector

from rnd.model import RND
from rnd.calib_band_with_demo import _get_band_modulation


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
    output_dir = str(pathlib.Path(rnd_ckpt).parent / 'calib' / 'rollout')
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
    base_cfg.policy.n_action_steps = n_action_steps
    base: FlowMatchUnetImagePolicy = hydra.utils.instantiate(base_cfg.policy)
    base.load_state_dict(base_payload['state_dicts']['ema_model'])
    base.to(device)
    base.eval()
    base.requires_grad_(False)


    # 2 load or rollout for rnd scores
    task = base_cfg.task_name
    cache_score_path = pathlib.Path(output_dir) / f'{task}_rnd_score.npy'
    if not pathlib.Path(cache_score_path).exists():
        try:
            print('Cache does not exist. Rollout!')
            env_cfg = deepcopy(base_cfg.task.env_runner)
            env_cfg._target_ = 'diffusion_policy.env_runner.robomimic_image_runner_with_detector.RobomimicImageRunnerWithDetector'
            env_cfg.n_train = 0
            env_cfg.n_train_vis = 0
            env_cfg.n_test = 250
            env_cfg.n_test_vis = 0
            env_cfg.test_start_seed = 1_000_000
            env_cfg.n_envs = 50
            env_cfg.n_action_steps = n_action_steps
            env_runner = hydra.utils.instantiate(env_cfg, output_dir=output_dir)

            test_log = env_runner.run_with_detector(policy=base, detector=rnd)
            success = np.logical_not(np.array(test_log['failure'], dtype=bool))  # (n_test,)
            all_scores = np.array(test_log['rnd_scores'], dtype=np.float32)  # (n_test, T_sub)
            scores = all_scores[success]  # (n_success, T_sub)

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
    ## 3.0 necessary constants
    Ta = n_action_steps
    To = base_cfg.n_obs_steps
    T = base_cfg.task.env_runner.max_steps
    n_trajs = scores.shape[0]  # number of trajectories
    n_mean_trajs = int(np.ceil(mean_ratio * n_trajs))  # number of trajectories for mean
    print(f"Num of trajs: {n_trajs}, Num of mean trajs: {n_mean_trajs}")

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
