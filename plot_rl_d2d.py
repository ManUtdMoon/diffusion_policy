import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import click
import dill
import torch


@click.command()
@click.option('-r', '--rl_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-x', '--x_lim', default=3.0)
def plot_d2d(rl_dir, device, x_lim):
    """
    Plots uncertainty (d2d) curves from evaluation data.
    """
    # 1. Data loading
    # 1.1 find all rb_obs_emb files
    rl_path = pathlib.Path(rl_dir)
    rb_obs_emb_files = sorted(list(rl_path.glob('rb_obs_emb*.npy')))
    if not rb_obs_emb_files:
        print(f"No rb_obs_emb*.npy files found in {rl_dir}")
        return

    # 1.2 load demo_obs_emb
    cfg_path = rl_path / '.hydra' / 'config.yaml'
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        ckpt_dir = pathlib.Path(cfg['online_task']['base_ckpt']).parent
        ckpt_name = pathlib.Path(cfg['online_task']['base_ckpt']).stem
        demo_emb_path = ckpt_dir / f'{ckpt_name}_obs_emb_action.pt'
        
        # compute rgb_emb_dim from the first file, assuming they are all the same
        tmp_obs_emb = torch.as_tensor(np.load(rb_obs_emb_files[0]))
        obs_emb_dim = tmp_obs_emb.shape[-1]

        obs_shape_meta = cfg['online_task']['shape_meta']['obs']
        low_dims = sum([obs_shape_meta[k]['shape'][0] for k in obs_shape_meta.keys() if len(obs_shape_meta[k]['shape']) == 1])
        rgb_emb_dim = obs_emb_dim - low_dims

    demo_emb = torch.load(open(demo_emb_path, 'rb'), pickle_module=dill)
    demo_rgb_emb = demo_emb['obs_emb'][..., -obs_emb_dim : -obs_emb_dim + rgb_emb_dim].to(device) # (N,di)

    for emb_file in rb_obs_emb_files:
        # 1.1 load buffer obs_emb
        obs_emb = torch.as_tensor(np.load(emb_file)).to(device)
        obs_emb = obs_emb.flatten(end_dim=-2)
        
        # extract rgb embedding
        obs_emb_rgb = obs_emb[..., :rgb_emb_dim]

        # 1.3 compute d2d
        d2d = torch.cdist(obs_emb_rgb, demo_rgb_emb, p=2).min(dim=1).values  # (B,N)
        d2d = d2d.cpu().numpy()

        # 2. plot distribution
        fig, ax = plt.subplots()
        ax.hist(d2d.flatten(), density=True, bins=30, alpha=0.5)
        ax.set_xlabel('d2d')
        ax.set_xlim(0, float(x_lim))
        ax.set_ylabel('Density')
        ax.set_title(f'd2d Distribution for {emb_file.stem}')
        ax.grid(True)

        # Save figure
        save_path = os.path.join(rl_dir, f'd2d_distribution_{emb_file.stem}.png')
        fig.savefig(save_path)
        print(f"Distribution plot saved to {save_path}")
        plt.close(fig)

if __name__ == '__main__':
    plot_d2d()