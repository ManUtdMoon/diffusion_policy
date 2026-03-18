import argparse
import random
from pathlib import Path

import dill
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import robomimic.utils.file_utils as FileUtils
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.robomimic.robomimic_image_wrapper import (
    RobomimicEarlyStopWrapper,
    RobomimicImageWrapper,
)
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.flow_match_vib_unet_image_policy import (
    FlowMatchVibUnetImagePolicy,
)
from diffusion_policy.policy.latent_policy import ResiduePolicy as ZResiduePolicy
from diffusion_policy.policy.latent_policy import SumPolicy as ZSumPolicy
from diffusion_policy.policy.residue_policy import ResiduePolicy
from diffusion_policy.policy.sum_policy import SumPolicy


plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True

SEED = 0
ENV_SEED = 10_000
N_ACTION_STEPS = 8
MAX_EPISODE_STEPS = 200
DATASET_PATH = "/media/datahub-2/ydj/robomimicv030/square/mh/image_v141_subset_abs.hdf5"
BASE_CKPT = "data/outputs/2025.12.04/21.27.40_train_flow_match_vib_unet_image_square_image/checkpoints/epoch=1000-score=0.420.ckpt"
RESRL_CKPT_DIR = "data/outputs/2026.03.18/17.48.27_train_online_robomimic_workspace_square_image/checkpoints"
ZPRL_CKPT_DIR = "data/outputs/2026.03.18/17.48.34_train_online_vib_robomimic_workspace_square_image/checkpoints"
OUTPUT_DIR = Path("data/plot/action")


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 3:
            return torch.device("cuda:3")
        return torch.device("cuda:0")
    return torch.device("cpu")


def plot_actions(base_action, res_action, sum_action, env_index=0):
    n_env, horizon, _ = base_action.shape
    assert 0 <= env_index < n_env

    base_env_action = base_action[env_index]
    sum_env_action = sum_action[env_index]
    res_env_action = res_action[env_index]

    fig, axes = plt.subplots(3, 2, figsize=(10, 2.2 * 3), dpi=150)
    dim_axis_map = {0: "X", 1: "Y", 2: "Z"}
    t = np.arange(horizon)

    for dim, ax in enumerate(axes[:, 0]):
        ax.plot(t, base_env_action[:, dim], label="base", color="b", linestyle="--")
        ax.plot(t, sum_env_action[:, dim], label="sum", color="r")
        ax.set_ylabel(dim_axis_map[dim])
    axes[0, 0].legend(loc="upper right")

    for dim, ax in enumerate(axes[:, 1]):
        ax.plot(t, res_env_action[:, dim], label="residue", color="g")
        ax.set_ylabel(dim_axis_map[dim])
    axes[0, 1].legend(loc="upper right")

    fig.tight_layout()
    return fig


def _compute_action_stats(action_seq):
    pos_seq = action_seq[0, :, :3]
    delta_pos = np.diff(pos_seq, axis=0)
    vel = delta_pos / 0.05
    acc = np.diff(vel, axis=0) / 0.05
    return {
        "mean_delta_position": np.linalg.norm(delta_pos, axis=-1).mean(),
        "mean_velocity": np.linalg.norm(vel, axis=-1).mean(),
        "mean_acceleration": np.linalg.norm(acc, axis=-1).mean(),
    }


def stat_action(action_seq, base_action_seq, step):
    action_stats = _compute_action_stats(action_seq)
    base_action_stats = _compute_action_stats(base_action_seq)
    print(f"Action statistics @ {step}")
    print("Policy mean delta position:", action_stats["mean_delta_position"])
    print("Policy mean velocity:", action_stats["mean_velocity"])
    print("Policy mean acceleration:", action_stats["mean_acceleration"])
    print("Base mean delta position:", base_action_stats["mean_delta_position"])
    print("Base mean velocity:", base_action_stats["mean_velocity"])
    print("Base mean acceleration:", base_action_stats["mean_acceleration"])


def undo_transform_action(action, rot_tf):
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        action = action.reshape((-1, 2, 10))

    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3 : 3 + d_rot]
    gripper = action[..., [-1]]
    rot = rot_tf.inverse(rot)
    transformed = np.concatenate([pos, rot, gripper], axis=-1)

    if raw_shape[-1] == 20:
        transformed = transformed.reshape((*raw_shape[:-1], 14))
    return transformed


def load_base_policy(device):
    base_payload = torch.load(open(BASE_CKPT, "rb"), pickle_module=dill)
    base_cfg = base_payload["cfg"]
    base_cfg.policy.n_action_steps = N_ACTION_STEPS

    base_policy: FlowMatchVibUnetImagePolicy = hydra.utils.instantiate(base_cfg.policy)
    base_policy.load_state_dict(base_payload["state_dicts"]["ema_model"])
    base_policy.eval()
    base_policy.requires_grad_(False)
    base_policy.to(device)
    print(f"Loaded base policy from {BASE_CKPT}")
    print(
        "To:",
        base_cfg.n_obs_steps,
        "Ta:",
        N_ACTION_STEPS,
        "do:",
        base_policy.obs_feature_dim,
        "dz:",
        base_policy.vib_latent_dim,
        "da:",
        base_cfg.shape_meta.action.shape[0],
    )
    return base_policy, base_cfg


def build_env(shape_meta, n_obs_steps, n_action_steps):
    env_meta = FileUtils.get_env_metadata_from_dataset(DATASET_PATH)
    env_meta["env_kwargs"]["use_object_obs"] = False
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

    def env_fn():
        robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta)
        robomimic_env.env.hard_reset = False
        return MultiStepWrapper(
            RobomimicImageWrapper(
                env=RobomimicEarlyStopWrapper(robomimic_env),
                shape_meta=shape_meta,
                init_state=None,
                render_obs_key="agentview_image",
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=MAX_EPISODE_STEPS,
            reward_agg_method="discounted_sum",
        )

    envs = SyncVectorEnv([env_fn])
    envs.seed(ENV_SEED)
    return envs


def build_resrl_policy(base_policy, base_cfg, device):
    ta_steps = N_ACTION_STEPS
    obs_dim = base_policy.obs_feature_dim
    action_dim = base_cfg.shape_meta.action.shape[0]
    flat_action_dim = ta_steps * action_dim

    res_policy = ResiduePolicy(
        obs_dim=obs_dim,
        action_dim=flat_action_dim,
        log_std_min=-10.0,
        log_std_max=2.0,
        gamma=0.99**N_ACTION_STEPS,
        tau=0.005,
        init_alpha=0.01,
        auto_alpha=True,
        res_scale=0.05,
        num_qs=2,
        num_subset=2,
    ).to(device)
    sum_policy = SumPolicy(
        res_scale=0.05,
        obs_emb_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=ta_steps,
        base_policy=base_policy,
        res_policy=res_policy,
    )
    return sum_policy, res_policy, obs_dim, ta_steps, action_dim


def build_zprl_policy(base_policy, base_cfg, device):
    to_steps = base_cfg.n_obs_steps
    ta_steps = N_ACTION_STEPS
    obs_dim = base_policy.obs_feature_dim
    latent_dim = base_policy.vib_latent_dim
    obs_chunk_dim = to_steps * obs_dim
    action_dim = base_cfg.shape_meta.action.shape[0]
    flat_action_dim = ta_steps * action_dim

    res_policy = ZResiduePolicy(
        obs_dim=obs_chunk_dim,
        z_dim=latent_dim,
        action_dim=flat_action_dim,
        log_std_min=-10.0,
        log_std_max=2.0,
        actor_input_type="obs_action",
        gamma=0.99**N_ACTION_STEPS,
        tau=0.005,
        init_alpha=0.01,
        auto_alpha=True,
        res_scale=0.2,
        num_qs=2,
        num_subset=2,
    ).to(device)
    sum_policy = ZSumPolicy(
        res_scale=0.2,
        base_policy=base_policy,
        res_policy=res_policy,
    )
    return sum_policy, res_policy


@torch.no_grad()
def eval_resrl(sum_policy, base_policy, envs, device, obs_dim, ta_steps, action_dim, rot_tf):
    action_seq = []
    naction_seq = []
    base_naction_seq = []
    base_action_seq = []
    res_naction_seq = []

    done = False
    obs_seq = envs.reset()
    obs_seq_tensor = dict_apply(obs_seq, lambda x: torch.from_numpy(x).to(device=device))
    base_dict = base_policy.predict_action(obs_seq_tensor)
    count = 0
    rewards = np.zeros(1, dtype=np.float32)

    while not done:
        obs_emb_tensor = base_dict["obs_emb"][:, -obs_dim:].detach()
        base_naction_tensor = base_dict["naction"].detach()
        base_naction_flat = base_naction_tensor.flatten(start_dim=1).cpu().numpy()

        sum_dict = sum_policy.predict_train_action(
            base_naction_tensor,
            obs_emb_tensor,
            res_mask=None,
        )
        sum_dict = dict_apply(sum_dict, lambda x: x.detach().cpu().numpy())
        action = sum_dict["action"]
        res_naction_flat = sum_dict["res_naction_flat"]

        env_action = undo_transform_action(action, rot_tf)
        next_obs_seq, rewards, dones, infos = envs.step(env_action)
        done = dones[0]

        action_seq.append(action)
        naction_seq.append(sum_dict["naction"].copy())
        base_action_seq.append(base_dict["action"].detach().cpu().numpy())
        base_naction_seq.append(base_naction_flat.reshape((-1, ta_steps, action_dim)))
        res_naction_seq.append(res_naction_flat.reshape((-1, ta_steps, action_dim)))
        count += N_ACTION_STEPS

        next_obs_seq_tensor = dict_apply(
            next_obs_seq, lambda x: torch.from_numpy(x).to(device=device)
        )
        base_dict = base_policy.predict_action(next_obs_seq_tensor)

    print("Counts:", count)
    print("success?", rewards[0] > 0.5)

    action_seq = np.concatenate(action_seq, axis=1)
    naction_seq = np.concatenate(naction_seq, axis=1)
    base_naction_seq = np.concatenate(base_naction_seq, axis=1)
    base_action_seq = np.concatenate(base_action_seq, axis=1)
    res_naction_seq = np.concatenate(res_naction_seq, axis=1)
    fig = plot_actions(base_action_seq, res_naction_seq, action_seq, env_index=0)

    return fig, {
        "action_seq": action_seq,
        "naction_seq": naction_seq,
        "base_naction_seq": base_naction_seq,
        "base_action_seq": base_action_seq,
        "res_naction_seq": res_naction_seq,
    }


@torch.no_grad()
def eval_zprl(sum_policy, base_policy, envs, device, rot_tf):
    action_seq = []
    naction_seq = []
    base_naction_seq = []
    base_action_seq = []

    done = False
    obs_seq = envs.reset()
    obs_seq_tensor = dict_apply(obs_seq, lambda x: torch.from_numpy(x).to(device=device))
    obs_emb_tensor = base_policy.encode_obs(obs_seq_tensor)
    _, z_mean, z_logvar, z = base_policy.vib_forward(obs_emb_tensor)
    obs_z = torch.cat([obs_emb_tensor, z_mean, z_logvar, z], dim=-1)
    count = 0
    rewards = np.zeros(1, dtype=np.float32)

    while not done:
        sum_dict = sum_policy.predict_train_action(obs_z)
        sum_dict = dict_apply(sum_dict, lambda x: x.detach())
        action = sum_dict["action"].cpu().numpy()

        env_action = undo_transform_action(action, rot_tf)
        next_obs_seq, rewards, dones, infos = envs.step(env_action)
        done = dones[0]

        sum_dict_np = dict_apply(sum_dict, lambda x: x.detach().cpu().numpy())
        action_seq.append(action)
        naction_seq.append(sum_dict_np["naction"].copy())
        base_action_seq.append(sum_dict_np["base_action"].copy())
        base_naction_seq.append(sum_dict_np["base_naction"].copy())
        count += N_ACTION_STEPS

        next_obs_seq_tensor = dict_apply(
            next_obs_seq, lambda x: torch.from_numpy(x).to(device=device)
        )
        next_obs_emb_tensor = base_policy.encode_obs(next_obs_seq_tensor).detach()
        _, next_z_mean, next_z_logvar, next_z = base_policy.vib_forward(next_obs_emb_tensor)
        obs_z = torch.cat([next_obs_emb_tensor, next_z_mean, next_z_logvar, next_z], dim=-1)

    print("Counts:", count)
    print("success?", rewards[0] > 0.5)

    action_seq = np.concatenate(action_seq, axis=1)
    naction_seq = np.concatenate(naction_seq, axis=1)
    base_naction_seq = np.concatenate(base_naction_seq, axis=1)
    base_action_seq = np.concatenate(base_action_seq, axis=1)
    res_action_seq = action_seq - base_action_seq
    fig = plot_actions(base_action_seq, res_action_seq, action_seq, env_index=0)

    return fig, {
        "action_seq": action_seq,
        "naction_seq": naction_seq,
        "base_naction_seq": base_naction_seq,
        "base_action_seq": base_action_seq,
        "res_action_seq": res_action_seq,
    }


def load_resrl_checkpoint(res_policy, step):
    if step > 0:
        ckpt_path = f"{RESRL_CKPT_DIR}/step={step}.ckpt"
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        res_policy.load_state_dict(payload["res_policy"])
        print(f"Loaded ResRL checkpoint from {ckpt_path}")
    else:
        print("Use random-initialized ResRL residue policy")


def load_zprl_checkpoint(res_policy, step):
    if step > 0:
        ckpt_path = f"{ZPRL_CKPT_DIR}/step={step}.ckpt"
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        res_policy.load_state_dict(payload["res_policy"])
        print(f"Loaded ZPRL checkpoint from {ckpt_path}")
    else:
        print("Use random-initialized ZPRL residue policy")


def save_outputs(mode, step, fig, eval_result):
    stem = f"{mode}_step={step}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUTPUT_DIR / f"{stem}.png"
    npz_path = OUTPUT_DIR / f"{stem}.npz"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    np.savez(
        npz_path,
        action_seq=eval_result["action_seq"],
        base_action_seq=eval_result["base_action_seq"],
    )
    print(f"Saved figure to {fig_path}")
    print(f"Saved action arrays to {npz_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["zprl", "resrl"])
    parser.add_argument("step", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()
    device = get_device()
    rot_tf = RotationTransformer("axis_angle", "rotation_6d")

    base_policy, base_cfg = load_base_policy(device)
    envs = build_env(
        shape_meta=base_cfg.shape_meta,
        n_obs_steps=base_cfg.n_obs_steps,
        n_action_steps=N_ACTION_STEPS,
    )

    if args.mode == "resrl":
        sum_policy, res_policy, obs_dim, ta_steps, action_dim = build_resrl_policy(
            base_policy, base_cfg, device
        )
        load_resrl_checkpoint(res_policy, args.step)
        fig, eval_result = eval_resrl(
            sum_policy=sum_policy,
            base_policy=base_policy,
            envs=envs,
            device=device,
            obs_dim=obs_dim,
            ta_steps=ta_steps,
            action_dim=action_dim,
            rot_tf=rot_tf,
        )
    else:
        sum_policy, res_policy = build_zprl_policy(base_policy, base_cfg, device)
        load_zprl_checkpoint(res_policy, args.step)
        fig, eval_result = eval_zprl(
            sum_policy=sum_policy,
            base_policy=base_policy,
            envs=envs,
            device=device,
            rot_tf=rot_tf,
        )

    envs.close()
    save_outputs(args.mode, args.step, fig, eval_result)
    stat_action(eval_result["action_seq"], eval_result["base_action_seq"], args.step)

    res_pos_seq = (eval_result["action_seq"] - eval_result["base_action_seq"])[0, :, :3]
    print("Mean residue action norm:", np.linalg.norm(res_pos_seq, axis=-1).mean())


if __name__ == "__main__":
    main()
