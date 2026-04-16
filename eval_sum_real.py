"""
Usage (flip):
python eval_sum_real.py \
    -c data/outputs/<date>/<time>_train_online_vib_real_workspace_flip/checkpoints/latest.ckpt \
    -o data/eval/flip/sum_real \
    -d cuda:0 \
    -t 16
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

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
import time

from diffusion_policy.env_runner.flip_runner import FlipRunner
from diffusion_policy.env_runner.juicing_runner import JuicingRunner
from diffusion_policy.env_runner.box_runner import BoxRunner
from diffusion_policy.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy
from diffusion_policy.policy.latent_policy import (
    ResiduePolicy as LatentResiduePolicy,
    SumPolicy as LatentSumPolicy,
)
from diffusion_policy.policy.residue_policy import ResiduePolicy as ActionResiduePolicy
from diffusion_policy.policy.sum_policy import SumPolicy as ActionSumPolicy


@click.command()
@click.option("-c", "--checkpoint", required=True, help="Online residual checkpoint path.")
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("-t", "--n_action_steps", default=16, type=int, required=True)
@click.option("-n", "--eval_episodes", default=30, type=int, show_default=True)
@click.option("-m", "--max_steps", default=250, type=int, show_default=True)
@click.option("-s", "--num_inference_steps", default=5, type=int, show_default=True)
def main(checkpoint, output_dir, device, n_action_steps, eval_episodes, max_steps, num_inference_steps):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    res_policy_state_dict = payload["res_policy"]
    To = int(cfg.n_obs_steps)
    Ta = int(n_action_steps)

    print(f"load RL ckpt @ step = {payload['global_step']} from {checkpoint}")

    seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # 1) base policy
    base_ckpt = cfg.online_task.base_ckpt
    base_payload = torch.load(open(base_ckpt, "rb"), pickle_module=dill)
    base_cfg = base_payload["cfg"]

    base_task_name = base_cfg.task_name
    online_task_name = cfg.task_name
    assert base_task_name == online_task_name, (
        f"Base policy task {base_task_name} does not match current task {online_task_name}"
    )

    # keep action horizon consistent with eval setting
    base_cfg.n_action_steps = Ta
    base_cfg.policy.n_action_steps = Ta
    base_cfg.task.dataset.pad_after = Ta - 1

    base_policy: FlowMatchVibUnetImagePolicy = hydra.utils.instantiate(base_cfg.policy)
    base_model_state = base_payload["state_dicts"]["ema_model"]
    base_policy.load_state_dict(base_model_state)
    base_policy.eval()
    base_policy.requires_grad_(False)
    base_policy.to(device)

    if hasattr(base_policy, "num_inference_steps"):
        base_policy.num_inference_steps = num_inference_steps
    if hasattr(base_policy, "n_action_steps"):
        base_policy.n_action_steps = Ta
    print(f"Loaded base policy from {base_ckpt}")

    # 2) residual + sum policy
    cfg.n_action_steps = Ta

    do = int(base_policy.obs_feature_dim)
    Do = int(To * do)
    da = int(cfg.shape_meta.action.shape[0])
    Da = int(Ta * da)
    res_target = str(cfg.res_policy._target_)

    if "latent_policy" in res_target:
        z_dim = int(base_policy.vib_latent_dim)
        res_policy: LatentResiduePolicy = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=Do, z_dim=z_dim, action_dim=Da
        )
        sum_policy = LatentSumPolicy(
            res_scale=cfg.training.res_scale,
            base_policy=base_policy,
            res_policy=res_policy,
        )
        print(
            f"Loaded latent residual policy with To={To}, do={do}, Do={Do}, "
            f"Ta={Ta}, da={da}, Da={Da}, z_dim={z_dim}"
        )
    else:
        res_policy: ActionResiduePolicy = hydra.utils.instantiate(
            cfg.res_policy, obs_dim=do, action_dim=Da
        )
        sum_policy = ActionSumPolicy(
            res_scale=cfg.training.res_scale,
            obs_emb_dim=do,
            action_dim=da,
            n_action_steps=Ta,
            base_policy=base_policy,
            res_policy=res_policy,
        )
        print(f"Loaded action residual policy with To={To}, do={do}, Ta={Ta}, da={da}, Da={Da}")

    res_policy.load_state_dict(res_policy_state_dict)
    res_policy.eval()
    res_policy.requires_grad_(False)
    res_policy.to(device)
    sum_policy.eval()

    # 3) run real-task eval (flip)
    mode = base_cfg.task.dataset.get("mode", None)
    key_epi_init = base_cfg.task.dataset.get("key_epi_init", None)
    env_runner = BoxRunner(
        output_dir=output_dir,
        eval_episodes=eval_episodes,
        max_steps=max_steps,
        n_obs_steps=To,
        n_action_steps=Ta,
        # mode=mode,
        # key_epi_init=key_epi_init,
    )
    runner_log = env_runner.run(sum_policy)

    # 4) dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)
    print(f"Saved eval log to {out_path}")


if __name__ == "__main__":
    main()
