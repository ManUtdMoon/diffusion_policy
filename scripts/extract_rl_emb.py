"""
For each specified RL step checkpoint, load (base_policy + res_policy) from
latent_policy.py, then pass 1000 samples through to extract:
  - obs_emb           : base obs embedding                      (N, Do)
  - modified_obs_emb  : VIB + res_z decoded embedding           (N, Do)
  - z_mean            : VIB latent mean                         (N, Dz)
  - res_z             : residual latent from res_policy         (N, Dz)
  - action            : unnormalised sum action                  (N, Ta, da)
  - naction           : normalised sum action                    (N, Ta*da)

Results are saved per step to:
  <output_dir>/step=<step>.pt

Usage:
  python scripts/extract_rl_emb.py \\
      --base-ckpt  /path/to/base.ckpt \\
      --rl-dir     /path/to/rl/run \\
      --steps      200000 400000 600000 \\
      --data       data/umap/square_image_abs_1000samples.pt \\
      --output-dir data/umap/rl_emb \\
      --device     cuda:0
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import argparse
import torch
import dill
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf

from diffusion_policy.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy
from diffusion_policy.policy.latent_policy import ResiduePolicy, SumPolicy, _agg_obs

OmegaConf.register_new_resolver("eval", eval, replace=True)

# ------------------------------------------------------------------ #
# loaders
# ------------------------------------------------------------------ #

def load_base_policy(ckpt_path: str, device: torch.device) -> FlowMatchVibUnetImagePolicy:
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    policy: FlowMatchVibUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][state_key])
    policy.eval()
    policy.requires_grad_(False)
    policy.to(device)
    print(f"Loaded base policy ({state_key}) from {ckpt_path}")
    return policy


def load_sum_policy(
        rl_ckpt_path: str,
        base_policy: FlowMatchVibUnetImagePolicy,
        device: torch.device,
) -> SumPolicy:
    payload = torch.load(open(rl_ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    To = cfg.n_obs_steps
    Ta = cfg.n_action_steps
    do = base_policy.obs_feature_dim
    Do = To * do
    dz = base_policy.vib_latent_dim
    da = cfg.shape_meta.action.shape[0]
    Da = Ta * da

    res_policy: ResiduePolicy = hydra.utils.instantiate(
        cfg.res_policy, obs_dim=Do, z_dim=dz, action_dim=Da)
    res_policy.load_state_dict(payload["res_policy"])
    res_policy.eval()
    res_policy.requires_grad_(False)
    res_policy.to(device)

    sum_policy = SumPolicy(
        res_scale=cfg.training.res_scale,
        base_policy=base_policy,
        res_policy=res_policy,
    )
    sum_policy.eval()
    print(f"  Do={Do}, Dz={dz}, Da={Da}, res_scale={cfg.training.res_scale}, "
          f"actor_input_type={res_policy.actor_input_type}")
    return sum_policy


# ------------------------------------------------------------------ #
# extraction
# ------------------------------------------------------------------ #

@torch.no_grad()
def extract(
        sum_policy: SumPolicy,
        samples: dict,
        batch_size: int,
        device: torch.device,
) -> dict:
    base_policy = sum_policy.base_policy
    res_policy  = sum_policy.res_policy
    Do, dz      = res_policy.obs_dim, res_policy.z_dim

    n = next(iter(samples["obs"].values())).shape[0]

    all_obs_emb          = []
    all_modified_obs_emb = []
    all_z_mean           = []
    all_res_z            = []
    all_action           = []
    all_naction          = []

    pbar = tqdm(total=n, desc="  extracting", unit="sample")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        obs_batch = {k: v[start:end].to(device) for k, v in samples["obs"].items()}

        # replicate SumPolicy.predict_action, keeping intermediates
        obs_emb            = base_policy.encode_obs(obs_batch)          # (B, Do)
        z_mean, z_logvar   = base_policy.vib_encoder(obs_emb)           # (B, Dz)
        obs_z              = torch.cat([obs_emb, z_mean, z_logvar, z_mean], dim=-1)

        res_input          = _agg_obs(sum_policy.actor_input_type, obs_z, Do, dz)
        res_z              = res_policy.predict_res_z(res_input, argmax=True)  # (B, Dz)

        perturbed_z        = z_mean + sum_policy.res_scale * res_z
        modified_obs_emb   = base_policy.vib_decoder(perturbed_z)       # (B, Do)

        result             = base_policy.conditional_predict(modified_obs_emb)
        action             = result["action"]    # (B, Ta, da)
        naction            = result["naction"]   # (B, Ta, da)

        all_obs_emb.append(obs_emb.cpu())
        all_modified_obs_emb.append(modified_obs_emb.cpu())
        all_z_mean.append(z_mean.cpu())
        all_res_z.append(res_z.cpu())
        all_action.append(action.cpu())
        all_naction.append(naction.reshape(end - start, -1).cpu())
        pbar.update(end - start)

    pbar.close()

    return {
        "obs_emb":           torch.cat(all_obs_emb,          dim=0),  # (N, Do)
        "modified_obs_emb":  torch.cat(all_modified_obs_emb, dim=0),  # (N, Do)
        "z_mean":            torch.cat(all_z_mean,           dim=0),  # (N, Dz)
        "res_z":             torch.cat(all_res_z,            dim=0),  # (N, Dz)
        "action":            torch.cat(all_action,           dim=0),  # (N, Ta, da)
        "naction":           torch.cat(all_naction,          dim=0),  # (N, Ta*da)
    }


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-dir",      required=True,
                        help="RL run directory containing checkpoints/step=*.ckpt")
    parser.add_argument("--steps",       required=True, nargs="+", type=int,
                        help="Step numbers to evaluate, e.g. --steps 200000 400000 600000")
    parser.add_argument("--data",        default="data/umap/square_image_abs_1000samples.pt")
    parser.add_argument("--output-dir",  default="data/umap/rl_emb")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--device",      default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    # read base_ckpt path from the first available RL checkpoint
    first_ckpt = os.path.join(args.rl_dir, "checkpoints", f"step={args.steps[0]}.ckpt")
    rl_payload  = torch.load(open(first_ckpt, "rb"), pickle_module=dill)
    base_ckpt   = rl_payload["cfg"]["online_task"]["base_ckpt"]
    print(f"Base policy path from RL config: {base_ckpt}")
    base_policy = load_base_policy(base_ckpt, device)

    print(f"Loading samples from {args.data} ...")
    samples = torch.load(args.data, map_location="cpu")
    n = next(iter(samples["obs"].values())).shape[0]
    print(f"  {n} samples loaded")

    os.makedirs(args.output_dir, exist_ok=True)

    for step in args.steps:
        ckpt_path = os.path.join(args.rl_dir, "checkpoints", f"step={step}.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"[WARN] not found: {ckpt_path}, skipping")
            continue

        print(f"\n── step={step} ──────────────────────────────")
        sum_policy = load_sum_policy(ckpt_path, base_policy, device)
        result = extract(sum_policy, samples, args.batch_size, device)

        # cosine sim: obs_emb vs modified_obs_emb
        e1 = torch.nn.functional.normalize(result["obs_emb"],         dim=-1)
        e2 = torch.nn.functional.normalize(result["modified_obs_emb"],dim=-1)
        cos_sim = (e1 * e2).sum(dim=-1)
        print(f"  cos_sim(obs_emb, modified_obs_emb): "
              f"mean={cos_sim.mean():.4f}  std={cos_sim.std():.4f}")
        print(f"  res_z rms: {result['res_z'].pow(2).mean().sqrt():.4f}")

        # compute rms
        delta_emb = result["modified_obs_emb"] - result["obs_emb"]
        delta_rms = delta_emb.pow(2).mean(dim=-1).sqrt().mean()
        base_rms = result["obs_emb"].pow(2).mean(dim=-1).sqrt().mean()
        mod_rms = result["modified_obs_emb"].pow(2).mean(dim=-1).sqrt().mean()

        print(f"  delta_emb RMS: {delta_rms:.4f}")
        print(f"  obs_emb RMS: {base_rms:.4f}")
        print(f"  modified_obs_emb RMS: {mod_rms:.4f}")
        print(f"  delta_emb / obs_emb RMS: {delta_rms / base_rms:.4f}")

        # compute recon error / euclidean distance
        recon_error = (delta_emb).pow(2).mean(dim=-1).sqrt().mean()
        print(f"  recon_error (euclidean distance): {recon_error:.4f}")

        out_path = os.path.join(args.output_dir, f"step={step}.pt")
        torch.save(result, out_path)
        print(f"  saved → {out_path}")
        print(f"  shapes: { {k: tuple(v.shape) for k, v in result.items()} }")
