"""
Load base and per-case embeddings, run UMAP on modified_obs_emb and naction,
and save results.

Directory layout expected:
  <data-dir>/base_recon.pt
  <data-dir>/<case>/step=<step>.pt

Output:
  <data-dir>/data.pkl      -- nested dict with numpy arrays
  <data-dir>/umap.pkl      -- UMAP projections (2D + 3D) for emb and naction

Usage:
  python scripts/1_umap_emb.py \\
      --data-dir  data/umap/rl_emb \\
      --cases     exp_a exp_b \\
      [--steps    100000 200000 400000 600000] \\
      [--n-neighbors 15] \\
      [--min-dist  0.1]
"""
import argparse
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import umap


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #

def load_pt(path: str, keys: list) -> dict:
    payload = torch.load(path, map_location="cpu")
    return {k: payload[k].numpy() for k in keys}


def nested_apply(d: dict, fn):
    """Recursively apply fn to all np.ndarray leaves."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = nested_apply(v, fn)
        else:
            out[k] = fn(v)
    return out


def collect_arrays(d: dict, key: str) -> list:
    """Collect all arrays for a given key, in DFS order. Returns list of (label, array)."""
    results = []
    for case_name, case_val in d.items():
        if isinstance(case_val, dict) and key in case_val:
            # base level: {base: {modified_obs_emb: ..., naction: ...}}
            results.append((case_name, case_val[key]))
        elif isinstance(case_val, dict):
            # case level: {case: {step: {modified_obs_emb: ..., naction: ...}}}
            for step, step_val in case_val.items():
                if isinstance(step_val, dict) and key in step_val:
                    results.append(((case_name, step), step_val[key]))
    return results


def run_umap(arrays: np.ndarray, n_components: int, **umap_kwargs) -> np.ndarray:
    reducer = umap.UMAP(n_components=n_components, **umap_kwargs)
    return reducer.fit_transform(arrays)


def split_projections(labels, proj: np.ndarray) -> dict:
    """Split a flat projection back into the nested dict shape."""
    result = {}
    idx = 0
    for label, arr in labels:
        n = arr.shape[0]
        chunk = proj[idx: idx + n]
        idx += n
        if isinstance(label, tuple):
            case_name, step = label
            result.setdefault(case_name, {})[step] = chunk
        else:
            result[label] = chunk
    return result


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    required=True,
                        help="Directory with base_recon.pt and <case>/step=*.pt")
    parser.add_argument("--cases",       required=True, nargs="+",
                        help="Case subdirectory names under data-dir")
    parser.add_argument("--steps",       nargs="+", type=int,
                        default=[100000, 200000, 400000, 600000])
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist",    type=float, default=0.1)
    parser.add_argument("--n-samples",   type=int, default=None,
                        help="Subsample N points per group for UMAP (default: use all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    KEYS = ["modified_obs_emb", "naction"]
    umap_kwargs = dict(n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    # ---- load data -------------------------------------------------------
    data = {}

    # base
    base_path = os.path.join(args.data_dir, "base_recon.pt")
    print(f"Loading base: {base_path}")
    data["base"] = load_pt(base_path, KEYS)

    # cases
    for case in args.cases:
        data[case] = {}
        for step in args.steps:
            pt_path = os.path.join(args.data_dir, case, f"step={step}.pt")
            if not os.path.exists(pt_path):
                print(f"  [WARN] not found: {pt_path}, skipping")
                continue
            print(f"  Loading {case}/step={step}")
            data[case][step] = load_pt(pt_path, KEYS)

    # ---- subsample -------------------------------------------------------
    if args.n_samples is not None:
        rng = np.random.RandomState(0)
        n_total = next(iter(data["base"].values())).shape[0]
        if args.n_samples < n_total:
            idx = rng.choice(n_total, size=args.n_samples, replace=False)
            idx.sort()
            print(f"\nSubsampling {args.n_samples}/{n_total} points")
            for group_key, group_val in data.items():
                if isinstance(group_val, dict) and any(isinstance(v, dict) for v in group_val.values()):
                    # case level: {step: {key: arr}}
                    for step in group_val:
                        for k in list(group_val[step].keys()):
                            group_val[step][k] = group_val[step][k][idx]
                else:
                    # base level: {key: arr}
                    for k in list(group_val.keys()):
                        group_val[k] = group_val[k][idx]

    # ---- UMAP ------------------------------------------------------------
    umap_results = {}

    for key in KEYS:
        labels_arrays = collect_arrays(data, key)
        flat = np.concatenate([arr for _, arr in labels_arrays], axis=0)
        flat_f32 = flat.astype(np.float32)

        print(f"\nRunning UMAP on '{key}'  shape={flat_f32.shape}")

        for n_components in [2, 3]:
            print(f"  → {n_components}D ...", flush=True)
            proj = run_umap(flat_f32, n_components=n_components, **umap_kwargs)
            umap_results[f"{key}_{n_components}d"] = split_projections(labels_arrays, proj)

    # ---- save ------------------------------------------------------------
    suffix = f"_n{args.n_samples}" if args.n_samples is not None else ""
    data_out   = os.path.join(args.data_dir, f"data{suffix}.pkl")
    umap_out   = os.path.join(args.data_dir, f"umap{suffix}.pkl")

    with open(data_out, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved data  → {data_out}")

    with open(umap_out, "wb") as f:
        pickle.dump(umap_results, f)
    print(f"Saved umap  → {umap_out}")

    # ---- summary ---------------------------------------------------------
    def _summarize(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                _summarize(v, prefix=f"{prefix}{k}/")
            else:
                print(f"  {prefix}{k}: {v.shape}")

    print("\n── data ──")
    _summarize(data)
    print("\n── umap ──")
    _summarize(umap_results)
