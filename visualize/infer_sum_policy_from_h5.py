if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)


import copy
from pathlib import Path

import click
import dill
import h5py
import numpy as np
import torch
import hydra
import yaml
from omegaconf import OmegaConf

from diffusion_policy.policy.latent_policy import ResiduePolicy, SumPolicy


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _load_run_cfg(traj_h5: Path):
    cfg_path = traj_h5.parent / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing run config: {cfg_path}")
    return OmegaConf.load(str(cfg_path)), cfg_path


def _resolve_base_ckpt(traj_h5: Path, base_ckpt: Path | None):
    if base_ckpt is not None:
        return Path(base_ckpt)
    cfg, _ = _load_run_cfg(traj_h5)
    ckpt = cfg.online_task.base_ckpt
    if ckpt is None:
        raise ValueError("online_task.base_ckpt missing in .hydra/config.yaml")
    return Path(ckpt)


@click.command()
@click.option(
    "--traj-h5",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True
)
@click.option(
    "--base-ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional override for online_task.base_ckpt"
)
@click.option("--n-pred-chunks", type=int, default=9, show_default=True)
@click.option("--device", type=str, default="cuda:0", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--pred-root",
    type=str,
    default="pred/random_sum",
    show_default=True,
    help="H5 group to store prediction result"
)
def main(traj_h5, base_ckpt, n_pred_chunks, device, seed, pred_root):
    traj_h5 = Path(traj_h5)
    run_cfg, cfg_path = _load_run_cfg(traj_h5)
    base_ckpt = _resolve_base_ckpt(traj_h5, base_ckpt)
    if not base_ckpt.exists():
        raise FileNotFoundError(f"base_ckpt not found: {base_ckpt}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    payload = torch.load(open(base_ckpt, "rb"), pickle_module=dill, map_location="cpu")
    base_cfg = copy.deepcopy(payload["cfg"])
    base_cfg.policy.n_action_steps = run_cfg.n_action_steps

    base_policy = hydra.utils.instantiate(base_cfg.policy)
    base_policy.load_state_dict(payload["state_dicts"]["ema_model"])
    base_policy.eval()
    base_policy.requires_grad_(False)

    To = int(run_cfg.n_obs_steps)
    Ta = int(run_cfg.n_action_steps)
    do = int(base_policy.obs_feature_dim)
    Do = To * do
    dz = int(base_policy.vib_latent_dim)
    da = int(run_cfg.shape_meta.action.shape[0])
    Da = Ta * da

    res_policy: ResiduePolicy = hydra.utils.instantiate(
        run_cfg.res_policy,
        obs_dim=Do,
        z_dim=dz,
        action_dim=Da
    )
    # Random init, no checkpoint load.
    res_policy.eval()

    sum_policy = SumPolicy(
        res_scale=float(run_cfg.training.res_scale),
        base_policy=base_policy,
        res_policy=res_policy
    )
    sum_policy.eval()

    device_t = torch.device(device)
    base_policy.to(device_t)
    res_policy.to(device_t)

    with h5py.File(traj_h5, "a") as f:
        obs_group = f["sparse/obs_seq"]
        obs_np = {k: obs_group[k][:] for k in obs_group.keys()}
        n_sparse = next(iter(obs_np.values())).shape[0]

        pred = np.zeros((n_sparse, n_pred_chunks, Ta, da), dtype=np.float32)
        with torch.no_grad():
            for i in range(n_sparse):
                obs_i = {}
                for k, v in obs_np.items():
                    obs_i[k] = torch.from_numpy(v[i:i+1]).to(device=device_t, dtype=torch.float32)
                # Match workspace exploration path:
                # obs -> obs_emb -> vib_forward -> obs_z -> predict_train_action(perturb=True)
                obs_emb = base_policy.encode_obs(obs_i)
                _, z_mean, z_logvar, z = base_policy.vib_forward(obs_emb, deterministic=False)
                obs_z = torch.cat([obs_emb, z_mean, z_logvar, z], dim=-1)  # (1, Do + 3*dz)

                # Batch-generate N stochastic chunks in one forward pass for efficiency.
                obs_z_batch = obs_z.repeat(n_pred_chunks, 1)  # (N, Do + 3*dz)
                out = sum_policy.predict_train_action(obs_z_batch, perturb=True)
                pred[i] = out["action"].detach().cpu().numpy().astype(np.float32)

        if f"{pred_root}/chunk_action" in f:
            del f[f"{pred_root}/chunk_action"]
        f.create_dataset(
            f"{pred_root}/chunk_action",
            data=pred,
            compression="gzip",
            compression_opts=4
        )
        g = f[pred_root]
        g.attrs["source"] = "random_res_policy_with_sum_policy_predict_train_action"
        g.attrs["cfg_path"] = str(cfg_path)
        g.attrs["base_ckpt"] = str(base_ckpt)
        g.attrs["n_pred_chunks"] = int(n_pred_chunks)
        g.attrs["seed"] = int(seed)
        g.attrs["device"] = str(device)

    print(f"Saved predictions: {traj_h5}:{pred_root}/chunk_action shape={pred.shape}")


if __name__ == "__main__":
    main()
