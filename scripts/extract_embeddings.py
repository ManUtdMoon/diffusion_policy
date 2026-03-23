"""
Load a FlowMatchVibUnetImagePolicy checkpoint, pass 1000 samples through the
network, and save:
  - obs_emb           : raw observation embedding (B, Do)
  - modified_obs_emb  : VIB-decoded embedding (B, Do)
  - action            : flattened action chunk (B, Ta*Da)
  - naction           : normalised action chunk (B, Ta*Da)   [for RL policy use]
  - z_mean            : VIB latent mean (B, Dz)
  - z_logvar          : VIB latent log-variance (B, Dz)
  - z                 : VIB latent sample (B, Dz)

Interface note:
  If you later want to attach a ResiduePolicy / SumPolicy, the obs_z tensor
  used by that policy is cat([obs_emb, z_mean, z_logvar, z], dim=-1).
  This concatenation is already stored as 'obs_z' in the output.
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import torch
import dill
import hydra
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from diffusion_policy.policy.flow_match_vib_unet_image_policy import FlowMatchVibUnetImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #

def load_base_policy(ckpt_path: str, device: torch.device) -> FlowMatchVibUnetImagePolicy:
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    policy: FlowMatchVibUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
    # prefer ema weights when available
    state_key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][state_key])
    policy.eval()
    policy.requires_grad_(False)
    policy.to(device)
    print(f"Loaded base policy from {ckpt_path} (weights: {state_key})")
    return policy


@torch.no_grad()
def extract(
        policy: FlowMatchVibUnetImagePolicy,
        samples: dict,
        batch_size: int,
        device: torch.device,
) -> dict:
    """
    Run encode_obs → vib_forward(deterministic=True) over all samples.

    Returns a dict of CPU tensors.
    """
    n = next(iter(samples["obs"].values())).shape[0]
    n_action_steps = policy.n_action_steps
    n_obs_steps    = policy.n_obs_steps

    all_obs_emb          = []
    all_modified_obs_emb = []
    all_z_mean           = []
    all_z_logvar         = []
    all_z                = []
    all_action           = []
    all_naction          = []

    pbar = tqdm(total=n, desc="Extracting embeddings", unit="sample")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        # --- obs batch ---
        obs_batch = {
            k: v[start:end].to(device)
            for k, v in samples["obs"].items()
        }

        # --- encode observation ---
        obs_emb = policy.encode_obs(obs_batch)          # (B, Do)

        # --- VIB forward (deterministic) ---
        modified_obs_emb, z_mean, z_logvar, z = policy.vib_forward(
            obs_emb, deterministic=True)                # all (B, *)

        # --- action: take the Ta steps starting at To-1 ---
        action_chunk = samples["action"][start:end]     # (B, T, Da)
        start_idx = n_obs_steps - 1
        end_idx   = start_idx + n_action_steps
        action_flat  = action_chunk[:, start_idx:end_idx].reshape(end - start, -1)  # (B, Ta*Da)

        # normalised action (useful for RL policy)
        naction_chunk = policy.normalizer["action"].normalize(
            action_chunk.to(device))
        naction_flat = naction_chunk[:, start_idx:end_idx].reshape(end - start, -1).cpu()

        all_obs_emb.append(obs_emb.cpu())
        all_modified_obs_emb.append(modified_obs_emb.cpu())
        all_z_mean.append(z_mean.cpu())
        all_z_logvar.append(z_logvar.cpu())
        all_z.append(z.cpu())
        all_action.append(action_flat.cpu())
        all_naction.append(naction_flat)
        pbar.update(end - start)

    pbar.close()
    obs_emb_cat          = torch.cat(all_obs_emb,          dim=0)
    modified_obs_emb_cat = torch.cat(all_modified_obs_emb, dim=0)
    z_mean_cat           = torch.cat(all_z_mean,           dim=0)
    z_logvar_cat         = torch.cat(all_z_logvar,         dim=0)
    z_cat                = torch.cat(all_z,                dim=0)

    # obs_z: the concatenation used by ResiduePolicy / SumPolicy
    obs_z_cat = torch.cat([obs_emb_cat, z_mean_cat, z_logvar_cat, z_cat], dim=-1)

    return {
        "obs_emb":           obs_emb_cat,           # (N, Do)
        "modified_obs_emb":  modified_obs_emb_cat,  # (N, Do)
        "z_mean":            z_mean_cat,            # (N, Dz)
        "z_logvar":          z_logvar_cat,          # (N, Dz)
        "z":                 z_cat,                 # (N, Dz)
        "obs_z":             obs_z_cat,             # (N, Do+3*Dz)  — for RL policy
        "action":            torch.cat(all_action,  dim=0),   # (N, Ta*Da)
        "naction":           torch.cat(all_naction, dim=0),   # (N, Ta*Da)
    }


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True,
                        help="Path to FlowMatchVibUnetImagePolicy checkpoint (.ckpt)")
    parser.add_argument("--data",   default="data/umap/square_image_abs_5000samples.pt",
                        help="Path to the sampled dataset .pt file")
    parser.add_argument("--output", default="data/umap/embeddings.pt",
                        help="Output .pt path")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:3")
    return parser.parse_args()


if __name__ == "__main__":
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    args = parse_args()
    device = torch.device(args.device)

    # load policy
    policy = load_base_policy(args.ckpt, device)

    # load samples
    print(f"Loading samples from {args.data} ...")
    samples = torch.load(args.data, map_location="cpu")

    # extract embeddings
    result = extract(policy, samples, batch_size=args.batch_size, device=device)

    # per-sample cosine similarity between obs_emb and modified_obs_emb
    e1 = torch.nn.functional.normalize(result["obs_emb"],          dim=-1)
    e2 = torch.nn.functional.normalize(result["modified_obs_emb"], dim=-1)
    cos_sim = (e1 * e2).sum(dim=-1)  # (N,)
    print(f"\nCosine similarity between obs_emb and modified_obs_emb (N={len(cos_sim)}):")
    print(f"  mean={cos_sim.mean().item():.4f}  std={cos_sim.std().item():.4f}"
          f"  min={cos_sim.min().item():.4f}  max={cos_sim.max().item():.4f}")

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

    # save
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(result, args.output)
    print(f"Saved to {args.output}")

    # sanity check
    print("\nShapes:")
    for k, v in result.items():
        print(f"  {k}: {tuple(v.shape)}")
