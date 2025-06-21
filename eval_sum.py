"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import random
import numpy as np
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json

from diffusion_policy.policy.sum_policy import SumPolicy
from diffusion_policy.policy.residue_policy import ResiduePolicy
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    device = torch.device(device)
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    res_policy_state_dict = payload['res_policy']
    ## deterministic mode
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    ## 1. base_policy
    base_policy: BaseImagePolicy = hydra.utils.instantiate(cfg.base_policy)
    base_payload = torch.load(open(cfg.online_task.base_ckpt, 'rb'), pickle_module=dill)
    base_cfg = base_payload['cfg']
    assert base_cfg.task_name == cfg.task_name, \
        f"Base policy task {base_cfg.task_name} does not match current task {cfg.task_name}"
    base_policy.load_state_dict(base_payload['state_dicts']['ema_model'])
    print(f"Loaded base policy from {cfg.online_task.base_ckpt}")
    base_policy.eval()
    base_policy.requires_grad_(False)
    base_policy.to(device)

    ## 2. res_policy
    obs_emb_dim = base_policy.obs_feature_dim # do, Do=To*do
    act_dim = cfg.shape_meta.action.shape[0] # da
    act_seq_dim = cfg.n_action_steps * act_dim # Da=Ta*da
    res_policy: ResiduePolicy = hydra.utils.instantiate(
        cfg.res_policy, obs_dim=obs_emb_dim, action_dim=act_seq_dim)
    print(f"Residue policy with obs_dim={obs_emb_dim}, action_dim={act_seq_dim}")
    res_policy.load_state_dict(res_policy_state_dict)
    res_policy.eval()
    res_policy.requires_grad_(False)
    res_policy.to(device)

    ## 3. sum_policy
    sum_policy = SumPolicy(
        res_scale=cfg.training.res_scale,
        obs_emb_dim=obs_emb_dim,
        action_dim=act_dim,
        n_action_steps=cfg.n_action_steps,
        base_policy=base_policy,
        res_policy=res_policy
    )
    sum_policy.eval()

    # run eval
    cfg.online_task.env_runner.n_train = 0
    cfg.online_task.env_runner.n_train_vis = 0
    cfg.online_task.env_runner.n_test = 50
    cfg.online_task.env_runner.n_test_vis = 10
    cfg.online_task.env_runner.test_start_seed = 100_000
    cfg.online_task.env_runner.n_envs = 25
    env_runner = hydra.utils.instantiate(
        cfg.online_task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(sum_policy)
    
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
