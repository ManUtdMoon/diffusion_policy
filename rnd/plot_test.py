#!/usr/bin/env python3

import sys
import os
import pathlib
import click
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import dill
from copy import deepcopy

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from rnd.model import RND

@click.command()
@click.option('--rnd_ckpt', '-c', required=True, help='Path to trained RND checkpoint')
@click.option('--indices', '-i', required=True, type=str, help='Comma-separated list of indices to plot')
@click.option('--mod_type', '-m', default='const', type=str)
@click.option('--confidence_interval', '-ci', default=0.95, type=float)
@click.option('--src', '-s', default='rollout', type=str)
def main(rnd_ckpt, indices, mod_type, confidence_interval, src):
    """
    Given a rnd_ckpt and index sequences, it loads the calibrated band and the i-th tested rnd_scores,
    then plot both lines in a figure with legends.
    """
    # 1. load stuff
    output_dir = pathlib.Path(rnd_ckpt).parent / 'test'
    calib_dir = pathlib.Path(rnd_ckpt).parent / 'calib'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1.1 load rnd and base policy config to get task info
    print(f"Loading RND checkpoint from {rnd_ckpt}")
    rnd_payload = torch.load(open(rnd_ckpt, 'rb'), pickle_module=dill)
    rnd_cfg = deepcopy(rnd_payload['config'])
    policy_ckpt = rnd_cfg['policy']
    base_payload = torch.load(open(policy_ckpt, 'rb'), pickle_module=dill)
    base_cfg = base_payload['cfg']
    task = base_cfg.task_name
    max_steps = base_cfg.task.env_runner.max_steps

    # 1.2 load band
    CI = confidence_interval
    calib_data_path = calib_dir / f'{src}' / f'{task}_calib_band_{CI}_{mod_type}.pkl'
    print(f"Loading calibrated band from {calib_data_path}")
    calib_data = pickle.load(open(calib_data_path, 'rb'))
    band_mean = calib_data['band_mean']
    bound = calib_data['bound']
    Ta = max_steps // len(band_mean)

    # 1.3 load scores
    scores_path = output_dir / f'{task}_rnd_scores_{CI}_{mod_type}.pkl'
    print(f"Loading scores from {scores_path}")
    scores_data = pickle.load(open(scores_path, 'rb'))
    rnd_scores = scores_data['rnd_scores']

    # 2. plot for each index
    indices_to_plot = [int(i) for i in indices.split(',')]
    x_ticks = np.arange(len(band_mean)) * Ta

    for i in indices_to_plot:
        if i >= len(rnd_scores):
            print(f"Warning: Index {i} is out of bounds. Max index is {len(rnd_scores) - 1}. Skipping.")
            continue

        plt.figure(figsize=(6, 6))
        
        # Plot calibrated band and the score
        plt.fill_between(x_ticks, bound, color='blue', alpha=0.1, label='Band Bound')
        plt.plot(x_ticks, band_mean, label='Band Mean', color='blue')
        plt.plot(x_ticks, rnd_scores[i], label=f'RND Score', color='red')

        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel('RND Score', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'rnd_scores_plot_index_{i}.png'
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.close()

if __name__ == "__main__":
    main()
