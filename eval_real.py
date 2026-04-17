"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import time

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.juicing_runner import JuicingRunner
from diffusion_policy.env_runner.flip_runner import FlipRunner
from diffusion_policy.env_runner.box_runner import BoxRunner
from diffusion_policy.env_runner.wallet_runner import WalletRunner


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-t', '--n_action_steps', default=16, type=int, required=True)
@click.option('-s', '--num_inference_steps', default=2, type=int, required=True)
def main(checkpoint, output_dir, device, n_action_steps, num_inference_steps):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cfg.n_action_steps = n_action_steps
    cfg.policy.n_action_steps = n_action_steps
    cfg.task.dataset.pad_after = n_action_steps - 1
    cfg.policy.num_inference_steps = num_inference_steps
    obs_step_indices = getattr(cfg, 'obs_step_indices', None)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # run eval
    env_runner = WalletRunner(
        output_dir=output_dir,
        eval_episodes=40,
        max_steps=1000,
        n_obs_steps=cfg.n_obs_steps,
        obs_step_indices=obs_step_indices,
        n_action_steps=cfg.n_action_steps,
    )
    runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
