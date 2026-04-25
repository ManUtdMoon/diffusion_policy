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
from omegaconf import OmegaConf
import wandb
import json

from zprl.policy.remote_policy import RemoteImagePolicy


# a patch due to uploaded checkpoints using a absolute specified dataset path.
DATASET_ROOT = "./data_local/robomimicv030/"
if not os.path.exists(DATASET_ROOT):
    raise ValueError(f"Dataset root {DATASET_ROOT} does not exist! Please set the correct path to robomimicv030/.")


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('--server_addr', default='tcp://127.0.0.1:5555')
@click.option('--timeout_ms', default=60000, type=int)
def main(checkpoint, output_dir, server_addr, timeout_ms):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg_yaml = OmegaConf.to_yaml(cfg)
    if 'diffusion_policy' in cfg_yaml:
        cfg = OmegaConf.create(cfg_yaml.replace('diffusion_policy', 'zprl'))

    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg.online_task.env_runner.n_train = 0
    cfg.online_task.env_runner.n_train_vis = 0
    cfg.online_task.env_runner.n_test = 100
    cfg.online_task.env_runner.n_test_vis = 5
    cfg.online_task.env_runner.test_start_seed = 100_000
    cfg.online_task.env_runner.n_envs = 50
    task_name = cfg.online_task.task_name
    dataset_type = cfg.online_task.dataset_type
    dataset_filename = os.path.basename(cfg.online_task.env_runner.dataset_path)
    dataset_path = os.path.join(DATASET_ROOT, task_name, dataset_type, dataset_filename)
    cfg.online_task.env_runner.dataset_path = dataset_path
    env_runner = hydra.utils.instantiate(
        cfg.online_task.env_runner,
        output_dir=output_dir)

    policy = RemoteImagePolicy(server_addr=server_addr, timeout_ms=timeout_ms)
    try:
        runner_log = env_runner.run(policy)
    finally:
        policy.close()

    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    json_log['policy_server_addr'] = server_addr
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    print(f"Saved eval log to {out_path}")


if __name__ == '__main__':
    main()
