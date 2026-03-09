#!/usr/bin/env python3
import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import random
import time
from pathlib import Path

import dill
import h5py
import hydra
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.box.box_env import BoxEnv
from diffusion_policy.env.box.franka_wrapper import FrankaWrapper
from diffusion_policy.env.box.robotiq_wrapper import RobotiqWrapper
from diffusion_policy.env.box.xarm_wrapper import XArmWrapper
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy
from diffusion_policy.policy.latent_policy import (
    ResiduePolicy as LatentResiduePolicy,
    SumPolicy as LatentSumPolicy,
)


# Set your residual checkpoint here (hard-coded on purpose).
CHECKPOINT_PATH = "data/outputs/box_zprl/2026.03.09/14.23.44_train_online_vib_real_workspace_box/checkpoints/latest.ckpt"

# Set your replay trajectory here (hard-coded on purpose).
DEMO_H5_PATH = "/data/hdd2/dongjie/box/data_teaser/demo_002.h5"

# Runtime defaults (no CLI).
DEVICE = "cuda:0"
N_ACTION_STEPS = 16
MAX_STEPS = 400
NUM_INFERENCE_STEPS = 2
REPLAY_HZ = 30.0

# Same initial joints as replay.py
REPLAY_XARM_JOINTS_INIT = [-34, -18.3, -61.9, -4.6, 81.5, -32.6]
REPLAY_FRANKA_JOINTS_INIT = (0.6649, -0.4903, 0.0028, -2.4483, -0.2356, 1.8972, 1.7677)


def _to_xarm_cartesian_cmd(xarm_target_m_rad: np.ndarray) -> np.ndarray:
    cmd = np.array(xarm_target_m_rad, dtype=np.float32).copy()
    cmd[:3] *= 1000.0  # m -> mm
    cmd[3:] = cmd[3:] * 180.0 / np.pi  # rad -> deg
    return cmd


def _load_demo_actions(demo_h5_path: Path) -> np.ndarray:
    with h5py.File(str(demo_h5_path), "r") as f:
        if "action" not in f:
            raise KeyError(f"'action' dataset not found in {demo_h5_path}")
        action = f["action"][:]
    if action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(f"Expected action shape (T, 14), got {action.shape}")
    return action


def load_sum_policy(
    checkpoint: str,
    device: torch.device,
    n_action_steps: int,
    num_inference_steps: int,
):
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    res_policy_state_dict = payload["res_policy"]
    To = int(cfg.n_obs_steps)
    Ta = int(n_action_steps)

    print(f"[Teaser] load residual ckpt @ step={payload['global_step']} from {checkpoint}")

    seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    base_ckpt = cfg.online_task.base_ckpt
    base_payload = torch.load(open(base_ckpt, "rb"), pickle_module=dill)
    base_cfg = base_payload["cfg"]

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

    do = int(base_policy.obs_feature_dim)
    da = int(cfg.shape_meta.action.shape[0])
    Da = int(Ta * da)
    Do = int(To * do)
    res_target = str(cfg.res_policy._target_)

    if "latent_policy" not in res_target:
        raise ValueError(f"Expected latent residual policy, but got: {res_target}")

    z_dim = int(base_policy.vib_latent_dim)
    res_policy: LatentResiduePolicy = hydra.utils.instantiate(
        cfg.res_policy, obs_dim=Do, z_dim=z_dim, action_dim=Da
    )
    sum_policy = LatentSumPolicy(
        res_scale=cfg.training.res_scale,
        base_policy=base_policy,
        res_policy=res_policy,
    )

    res_policy.load_state_dict(res_policy_state_dict)
    res_policy.eval()
    res_policy.requires_grad_(False)
    res_policy.to(device)
    sum_policy.eval()

    return sum_policy, To, Ta


@torch.no_grad()
def rollout_policy_once(policy, n_obs_steps: int, n_action_steps: int, max_steps: int):
    env = MultiStepWrapper(
        BoxEnv(),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
        reward_agg_method="sum",
    )

    try:
        obs = env.reset()
        policy.reset()
        done = False
        actual_step_count = 0
        episode_return = 0.0
        pre_reward = -1.0
        t_start = time.time()

        print("[Teaser] Segment-1 start: rollout composite policy until done.")
        while not done:
            t_frame = time.time()

            obs_dict_input = {
                "global": obs["global"].astype(np.float32),
                "wrist_0": obs["wrist_0"].astype(np.float32),
                "wrist_1": obs["wrist_1"].astype(np.float32),
                "qpos": obs["qpos"].astype(np.float32),
            }
            obs_dict = dict_apply(
                obs_dict_input,
                lambda x: torch.from_numpy(x).to(device=policy.device).unsqueeze(0),
            )
            action_dict = policy.predict_action(obs_dict)
            action = action_dict["action"].detach().cpu().numpy().squeeze(0)

            obs, reward, done, info = env.step(action.copy())

            if reward is None:
                reward = pre_reward
            episode_return += reward
            pre_reward = reward
            actual_step_count += 1

            print(f"[Teaser] Segment-1 chunk freq: {1.0 / max(1e-6, (time.time() - t_frame)):.2f} Hz")

        t_end = time.time()
        success = bool(info["is_success"][-1]) if "is_success" in info else False
        print(
            "[Teaser] Segment-1 done. "
            f"success={success}, return={episode_return:.3f}, "
            f"avg_chunk_freq={actual_step_count / max(1e-6, (t_end - t_start)):.2f} Hz"
        )
        env.reset_end()
    finally:
        env.close()


def replay_demo_segment(demo_h5_path: Path, dt: float):
    actions = _load_demo_actions(demo_h5_path)
    start_idx = 0
    end_idx = actions.shape[0]

    xarm = None
    franka = None
    xarm_gripper = None
    franka_gripper = None
    try:
        print("[Teaser] Segment-2 start: move to replay init joints and replay demo.")
        xarm = XArmWrapper(joints_init=REPLAY_XARM_JOINTS_INIT)
        xarm_gripper = RobotiqWrapper(robot="xarm")
        franka = FrankaWrapper(joints_init=REPLAY_FRANKA_JOINTS_INIT)
        franka_gripper = RobotiqWrapper(robot="franka")

        print(f"[Teaser] Replay steps [{start_idx}, {end_idx}) at {1.0 / dt:.2f} Hz")
        for i in range(start_idx, end_idx):
            step_start = time.time()
            a = actions[i]

            xarm_target_cmd = _to_xarm_cartesian_cmd(a[:6])
            xarm.set_servo_cartesian(xarm_target_cmd)
            if a[6] > 0.5:
                xarm_gripper.close()
            else:
                xarm_gripper.open()

            franka_target = np.array(a[7:13], dtype=np.float32)
            franka.franka.schedule_waypoint(franka_target, time.time() + dt)
            if a[13] > 0.5:
                franka_gripper.close()
            else:
                franka_gripper.open()

            while time.time() - step_start < dt:
                time.sleep(dt / 20)

            if (i - start_idx + 1) % 20 == 0 or i == end_idx - 1:
                print(f"[Teaser] Replay step {i + 1}/{end_idx}")

        print("[Teaser] Segment-2 done.")
    finally:
        xarm_gripper.open()
        franka_gripper.open()
        time.sleep(1.0)  # wait for grippers to open before shutdown
        if xarm_gripper is not None:
            xarm_gripper.shutdown()
        if franka_gripper is not None:
            franka_gripper.shutdown()
        if xarm is not None:
            xarm.close()
        if franka is not None:
            franka.close()


def main():
    checkpoint = Path(CHECKPOINT_PATH).expanduser()
    demo_h5_path = Path(DEMO_H5_PATH).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            f"Please edit CHECKPOINT_PATH in {Path(__file__).name}."
        )
    if not demo_h5_path.exists():
        raise FileNotFoundError(
            f"Demo h5 not found: {demo_h5_path}\n"
            f"Please edit DEMO_H5_PATH in {Path(__file__).name}."
        )
    if REPLAY_HZ <= 0:
        raise ValueError(f"REPLAY_HZ must be > 0, got {REPLAY_HZ}")

    torch_device = torch.device(DEVICE)
    sum_policy, To, Ta = load_sum_policy(
        checkpoint=str(checkpoint),
        device=torch_device,
        n_action_steps=N_ACTION_STEPS,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )

    rollout_policy_once(
        policy=sum_policy,
        n_obs_steps=To,
        n_action_steps=Ta,
        max_steps=MAX_STEPS,
    )

    replay_demo_segment(
        demo_h5_path=demo_h5_path,
        dt=1.0 / REPLAY_HZ,
    )


if __name__ == "__main__":
    main()
