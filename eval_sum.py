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

from zprl.policy.base_image_policy import BaseImagePolicy
from zprl.policy.latent_policy import (
    ResiduePolicy as LatentResiduePolicy,
    SumPolicy as LatentSumPolicy,
)
from zprl.policy.residue_policy import (
    ResiduePolicy as ActionResiduePolicy,
    SumPolicy as ActionSumPolicy,
)


# a patch due to uploaded checkpoints using a absolute specified dataset path.
DATASET_ROOT = "./data_local/robomimicv030/"
if not os.path.exists(DATASET_ROOT):
    raise ValueError(f"Dataset root {DATASET_ROOT} does not exist! Please set the correct path to robomimicv030/.")


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-b', '--base_ckpt', default=None, type=str)
def main(checkpoint, output_dir, device, base_ckpt):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    device = torch.device(device)
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    res_policy_state_dict = payload['res_policy']

    Ta = int(cfg.n_action_steps)
    To = int(cfg.n_obs_steps)

    ## deterministic mode
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if (base_ckpt is None) and not pathlib.Path(cfg.online_task.base_ckpt).exists():
        raise ValueError(
            f"Base policy not specified and not found at {cfg.online_task.base_ckpt}. Maybe downloading the checkpoint from internet? Please also download the base ckpt and specify the correct path using --base_ckpt.")
    elif base_ckpt is not None:
        cfg.online_task.base_ckpt = base_ckpt
        print(f"Using base policy from cli argument: {base_ckpt}")
        print("Make sure the RL policy matches the base policy for correct evaluation!")

    ## 1. base_policy
    base_payload = torch.load(open(cfg.online_task.base_ckpt, 'rb'), pickle_module=dill)
    base_cfg = base_payload['cfg']
    assert base_cfg.task_name == cfg.task_name, \
        f"Base policy task {base_cfg.task_name} does not match current task {cfg.task_name}"
    base_cfg.n_action_steps = Ta
    base_cfg.policy.n_action_steps = Ta
    base_cfg.task.env_runner.n_action_steps = Ta
    base_cfg.task.dataset.pad_after = Ta - 1
    base_policy: BaseImagePolicy = hydra.utils.instantiate(base_cfg.policy)
    base_policy.load_state_dict(base_payload['state_dicts']['ema_model'])
    print(f"Loaded base policy from {cfg.online_task.base_ckpt}")
    base_policy.eval()
    base_policy.requires_grad_(False)
    base_policy.to(device)

    ## 2. res_policy + sum_policy
    cfg.n_action_steps = Ta
    do = int(base_policy.obs_feature_dim)
    Do = int(To * do)
    da = int(cfg.shape_meta.action.shape[0])
    Da = int(Ta * da)
    res_target = str(cfg.res_policy._target_)

    if "latent_policy" in res_target:
        z_dim = int(base_policy.vib_latent_dim)
        res_policy = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=Do, z_dim=z_dim, action_dim=Da)
        sum_policy = LatentSumPolicy(
            res_scale=cfg.training.res_scale,
            base_policy=base_policy,
            res_policy=res_policy,
        )
        print(f"ZPRL (latent) with To={To}, do={do}, Do={Do}, "
              f"Ta={Ta}, da={da}, Da={Da}, z_dim={z_dim}")
    elif "residue_policy" in res_target:
        res_policy = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=do, action_dim=Da)
        sum_policy = ActionSumPolicy(
            res_scale=cfg.training.res_scale,
            obs_emb_dim=do,
            action_dim=da,
            n_action_steps=Ta,
            base_policy=base_policy,
            res_policy=res_policy,
        )
        print(f"ResRL (action) with To={To}, do={do}, Ta={Ta}, da={da}, Da={Da}")
    else:
        raise ValueError(f"Unknown res_policy target: {res_target}")

    res_policy.load_state_dict(res_policy_state_dict)
    res_policy.eval()
    res_policy.requires_grad_(False)
    res_policy.to(device)
    sum_policy.eval()

    # run eval
    cfg.online_task.env_runner.n_train = 0
    cfg.online_task.env_runner.n_train_vis = 0
    cfg.online_task.env_runner.n_test = 100
    cfg.online_task.env_runner.n_test_vis = 5
    cfg.online_task.env_runner.test_start_seed = 100_000
    cfg.online_task.env_runner.n_envs = 50
    # reset dataset_path to avoid location mismatch between training and evaluation.
    # extract the filename from the original config to avoid hardcoding (e.g., tool_hang uses a different name).
    task_name = cfg.online_task.task_name
    dataset_type = cfg.online_task.dataset_type
    dataset_filename = os.path.basename(cfg.online_task.env_runner.dataset_path)
    dataset_path = os.path.join(DATASET_ROOT, task_name, dataset_type, dataset_filename)
    cfg.online_task.env_runner.dataset_path = dataset_path
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
    print(f"Saved eval log to {out_path}")

if __name__ == '__main__':
    main()
