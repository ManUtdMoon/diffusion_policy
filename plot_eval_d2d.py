import json
import numpy as np
import matplotlib.pyplot as plt
import os
import click

@click.command()
@click.option('--eval_dir', required=True)
@click.option('-x', '--x_lim', default=3.0)
def plot_d2d(eval_dir, x_lim):
    """
    Plots uncertainty (d2d) curves from evaluation data.
    """
    # Data loading
    with open(os.path.join(eval_dir, 'eval_log.json'), 'r') as f:
        eval_log = json.load(f)
    d2d = np.load(os.path.join(eval_dir, 'uncertainty.npy'))

    # 1. plot curves
    # 1.1 Data processing
    failures = np.array(eval_log['failure'], dtype=bool)
    Ta = eval_log['n_action_steps']
    n_traj, n_step = d2d.shape
    t = np.arange(n_step) * Ta

    fig, ax = plt.subplots()
    for i, traj in enumerate(d2d[failures]):
        ax.plot(t, traj, label='Failure' if i == 0 else '', c='r', linewidth=1)
    for i, traj in enumerate(d2d[~failures]):
        ax.plot(t, traj, label='Success' if i == 0 else '', c='b', linewidth=1)

    ax.set_xlabel('Time')
    ax.set_ylabel('d2d')
    ax.set_title('d2d vs. Time')
    ax.legend()
    ax.grid(True)

    # Save figure
    save_path = os.path.join(eval_dir, 'd2d_vs_time.png')
    fig.savefig(save_path)
    print(f"Line plot saved to {save_path}")

    # 2. plot distribution
    fig, ax = plt.subplots()
    ax.hist(d2d.flatten(), density=True, bins=30, alpha=0.5)
    ax.set_xlabel('d2d')
    ax.set_xlim(0, float(x_lim))
    ax.set_ylabel('Density')
    ax.set_title('d2d Distribution')
    ax.grid(True)

    # Save figure
    save_path = os.path.join(eval_dir, 'd2d_distribution.png')
    fig.savefig(save_path)
    print(f"Distribution plot saved to {save_path}")

if __name__ == '__main__':
    plot_d2d()
