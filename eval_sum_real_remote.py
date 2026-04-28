"""
Usage (wallet):
python eval_sum_real_remote.py \
    -c data/outputs/<date>/<time>_train_online_vib_real_workspace_wallet/checkpoints/latest.ckpt \
    -o data/eval/wallet/sum_real_remote \
    -t 24 \
    --server_addr tcp://127.0.0.1:5555
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import json
import os
import pathlib
import time

import click
import dill
import numpy as np
import torch
import wandb

from diffusion_policy.policy.remote_policy import RemoteImagePolicy


@click.command()
@click.option("-c", "--checkpoint", required=True, help="Online residual checkpoint path.")
@click.option("-o", "--output_dir", required=True)
@click.option("-t", "--n_action_steps", default=24, type=int, required=True)
@click.option("-n", "--eval_episodes", default=3, type=int, show_default=True)
@click.option("-m", "--max_steps", default=1000, type=int, show_default=True)
@click.option("--server_addr", default="tcp://127.0.0.1:5555", show_default=True)
@click.option("--timeout_ms", default=60000, type=int, show_default=True)
def main(checkpoint, output_dir, n_action_steps, eval_episodes, max_steps, server_addr, timeout_ms):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    Ta = int(n_action_steps)

    base_ckpt = cfg.online_task.base_ckpt
    base_payload = torch.load(open(base_ckpt, "rb"), pickle_module=dill)
    base_cfg = base_payload["cfg"]
    To = int(base_cfg.n_obs_steps)
    obs_step_indices = getattr(base_cfg, "obs_step_indices", None)
    if obs_step_indices is not None:
        obs_step_indices = list(obs_step_indices)

    base_task_name = base_cfg.task_name
    online_task_name = cfg.task_name
    assert base_task_name == online_task_name, (
        f"Base policy task {base_task_name} does not match current task {online_task_name}"
    )
    if online_task_name != "wallet":
        raise ValueError(f"Step1 remote real eval currently supports wallet only, got {online_task_name}")

    if obs_step_indices is not None:
        print(
            "Using sparse obs history from base checkpoint: "
            f"obs_step_indices={obs_step_indices}, "
            f"env_n_obs_steps={max(obs_step_indices) + 1}, "
            f"policy_n_obs_steps={To}"
        )

    from diffusion_policy.env_runner.wallet_runner import WalletRunner

    env_runner = WalletRunner(
        output_dir=output_dir,
        eval_episodes=eval_episodes,
        max_steps=max_steps,
        n_obs_steps=To,
        obs_step_indices=obs_step_indices,
        n_action_steps=Ta,
    )

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
    json_log["policy_server_addr"] = server_addr
    json_log["n_action_steps"] = Ta
    json_log["n_obs_steps"] = To
    latencies = np.asarray(policy.predict_latencies_sec, dtype=np.float64)
    if latencies.size > 0:
        json_log["policy_latency_mean_sec"] = float(np.mean(latencies))
        json_log["policy_latency_max_sec"] = float(np.max(latencies))
        json_log["policy_latency_p95_sec"] = float(np.percentile(latencies, 95))
        json_log["policy_latency_count"] = int(latencies.size)

    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)
    print(f"Saved eval log to {out_path}")


if __name__ == "__main__":
    main()
