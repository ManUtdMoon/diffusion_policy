"""
Compute and plot metrics that quantify how each RL case's modified_obs_emb
and naction drift from the base, across training steps.

Metrics:
  1. Per-sample L2 displacement  (mean ± std)
  2. Per-sample cosine similarity (mean ± std)
  3. k-NN neighbourhood preservation  (fraction of base k-NN preserved)
  4. Per-dimension action displacement  (RMS per action dim)
  5. Perturbation-to-prior ratio  (||scale*res_z|| / ||std(z)||)
  6. Mahalanobis OOD score  (squared Mahalanobis of modified_obs_emb w.r.t. base)

Directory layout expected (same as 1_umap_emb.py):
  <data-dir>/base_recon.pt
  <data-dir>/<case>/step=<step>.pt

Usage:
  python scripts/3_metrics.py \\
      --data-dir   data/umap \\
      --cases      scale01 scale02 scale05 \\
      --scales     0.1 0.2 0.5 \\
      --output-dir data/umap/metrics \\
      [--steps     100000 200000 400000 600000] \\
      [--knn-k     10]
"""
import argparse
import os
import pickle
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ------------------------------------------------------------------ #
# font setup
# ------------------------------------------------------------------ #
FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ARIAL_PATH = os.path.abspath(os.path.join(FONT_DIR, "Arial.ttf"))
CAMBRIA_PATH = os.path.abspath(os.path.join(FONT_DIR, "cambria-math.ttf"))

fm.fontManager.addfont(ARIAL_PATH)
fm.fontManager.addfont(CAMBRIA_PATH)

ARIAL_NAME = fm.FontProperties(fname=ARIAL_PATH).get_name()
CAMBRIA_NAME = fm.FontProperties(fname=CAMBRIA_PATH).get_name()

plt.rcParams.update({
    "font.family": ARIAL_NAME,
    "font.size": 14,
    "mathtext.fontset": "custom",
    "mathtext.rm": CAMBRIA_NAME,
    "mathtext.it": CAMBRIA_NAME,
    "mathtext.bf": CAMBRIA_NAME,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
})

plt.rcParams.update({
    "font.family": ARIAL_NAME,
    "font.size": 14,
    "pdf.fonttype": 42,       # TrueType in PDF
    "ps.fonttype": 42,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

# colour palette for cases
CASE_COLORS = [
    "#e41a1c", "#ff7f00", "#4daf4a", "#984ea3",
    "#377eb8", "#a65628", "#f781bf", "#999999",
]

CASE_COLORS = [
    '#e8cd81', '#c66a42', '#6a4a2b',
]

CASE_NAMES = {
    "recon-s0.1": r"$\lambda=0.1$",
    "recon-s0.2": r"$\lambda=0.2$",
    "recon-s0.5": r"$\lambda=0.5$",
}

# ------------------------------------------------------------------ #
# data loading
# ------------------------------------------------------------------ #
METRIC_KEYS = ["modified_obs_emb", "naction"]  # keys used for standard metrics
# additional keys loaded when available (for perturbation ratio / OOD)
EXTRA_KEYS  = ["z_mean", "z_logvar", "res_z", "obs_emb"]


def load_pt(path: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    out = {}
    for k in METRIC_KEYS + EXTRA_KEYS:
        if k in payload:
            out[k] = payload[k].numpy() if isinstance(payload[k], torch.Tensor) else payload[k]
    return out


def load_all(data_dir, cases, steps):
    data = {}
    base_path = os.path.join(data_dir, "base_recon.pt")
    print(f"Loading base: {base_path}")
    data["base"] = load_pt(base_path)
    print(f"  base keys: {list(data['base'].keys())}")
    for case in cases:
        data[case] = {}
        for step in steps:
            pt_path = os.path.join(data_dir, case, f"step={step}.pt")
            if not os.path.exists(pt_path):
                print(f"  [WARN] not found: {pt_path}, skipping")
                continue
            data[case][step] = load_pt(pt_path)
            print(f"  loaded {case}/step={step}")
    return data


# ------------------------------------------------------------------ #
# metric functions — all operate on (N, D) numpy arrays
# ------------------------------------------------------------------ #

def l2_displacement(base: np.ndarray, case: np.ndarray):
    """Per-sample L2 distance.  Returns (N,) array."""
    return np.linalg.norm(case - base, axis=-1)


def cosine_similarity(base: np.ndarray, case: np.ndarray):
    """Per-sample cosine similarity.  Returns (N,) array."""
    dot = (base * case).sum(axis=-1)
    norm = np.linalg.norm(base, axis=-1) * np.linalg.norm(case, axis=-1)
    return dot / np.maximum(norm, 1e-8)


def perturbation_to_prior_ratio(z_logvar_base: np.ndarray,
                                res_z_case: np.ndarray,
                                scale: float):
    """
    Per-sample ratio:  ||scale * res_z|| / ||std(z_prior)||.
    Values >> 1 mean the perturbation exceeds the VIB decoder's training range.
    Returns (N,) array.
    """
    perturbation_norm = np.linalg.norm(scale * res_z_case, axis=-1)  # (N,)
    z_std = np.sqrt(np.exp(z_logvar_base))                           # (N, Dz)
    prior_norm = np.linalg.norm(z_std, axis=-1)                      # (N,)
    return perturbation_norm / np.maximum(prior_norm, 1e-8)          # (N,)


def mahalanobis_ood(base_emb: np.ndarray, case_emb: np.ndarray):
    """
    Squared Mahalanobis distance of each case sample w.r.t. the Gaussian
    fitted on base_emb (Ledoit-Wolf shrinkage for stability).
    Returns (N,) array.
    """
    cov_model = LedoitWolf().fit(base_emb)
    return np.sqrt(cov_model.mahalanobis(case_emb))  # (N,) squared distances


def knn_preservation(base: np.ndarray, case: np.ndarray, k=10):
    """Fraction of base k-NN preserved in case, per sample. Returns (N,)."""
    nn_base = NearestNeighbors(n_neighbors=k + 1).fit(base)
    nn_case = NearestNeighbors(n_neighbors=k + 1).fit(case)
    # +1 because query point is its own neighbour
    _, idx_base = nn_base.kneighbors(base)
    _, idx_case = nn_case.kneighbors(case)
    # drop self-neighbour (first column)
    idx_base = idx_base[:, 1:]
    idx_case = idx_case[:, 1:]
    overlap = np.array([
        len(set(b) & set(c)) / k
        for b, c in zip(idx_base, idx_case)
    ])
    return overlap  # (N,)


def per_dim_rms(base: np.ndarray, case: np.ndarray):
    """RMS displacement per dimension.  Returns (D,) array."""
    delta = case - base
    return np.sqrt((delta ** 2).mean(axis=0))


# ------------------------------------------------------------------ #
# plotting helpers
# ------------------------------------------------------------------ #

def new_fig(nrows=1, ncols=1, figsize=None):
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


def save_fig(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def case_color(idx):
    return CASE_COLORS[idx % len(CASE_COLORS)]


# ------------------------------------------------------------------ #
# main plotting routines
# ------------------------------------------------------------------ #

def plot_scalar_vs_step(metric_dict, steps, case_names, ylabel, title, out_path):
    """
    metric_dict[case][step] = (mean, std)
    Plots mean ± std vs step for each case.
    """
    fig, ax = new_fig()
    for ci, case in enumerate(case_names):
        xs, ys, errs = [], [], []
        for s in steps:
            if s in metric_dict.get(case, {}):
                xs.append(s * 4 / 1e6)
                m, sd = metric_dict[case][s]
                ys.append(m)
                errs.append(sd)
        if xs:
            ax.errorbar(xs, ys, yerr=errs, label=CASE_NAMES[case],
                        color=case_color(ci), capsize=3, marker="o", markersize=4)
    ax.set_xlabel("Env Steps (M)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_path)


def plot_median_iqr_vs_step(metric_dict, steps, case_names, ylabel, title, out_path):
    """
    metric_dict[case][step] = (median, q1, q3)
    Plots median with IQR error bars vs step for each case.
    """
    fig, ax = new_fig()
    for ci, case in enumerate(case_names):
        xs, meds, lo, hi = [], [], [], []
        for s in steps:
            if s in metric_dict.get(case, {}):
                xs.append(s * 4 / 1e6)
                median, q1, q3 = metric_dict[case][s]
                meds.append(median)
                lo.append(median - q1)   # distance below median
                hi.append(q3 - median)   # distance above median
        if xs:
            ax.errorbar(xs, meds, yerr=[lo, hi], label=CASE_NAMES.get(case, case),
                        color=case_color(ci), capsize=3, marker="o", markersize=4)
    ax.set_xlabel("Env steps (M)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_path)


def plot_per_dim_action(dim_dict, steps, case_names, out_path):
    """
    dim_dict[case][step] = (D,) RMS per dim.
    One subplot per step, grouped bar chart.
    """
    fig, axes = new_fig(1, len(steps), figsize=(4 * len(steps), 4))
    if len(steps) == 1:
        axes = [axes]
    for si, step in enumerate(steps):
        ax = axes[si]
        width = 0.8 / max(len(case_names), 1)
        for ci, case in enumerate(case_names):
            if step not in dim_dict.get(case, {}):
                continue
            vals = dim_dict[case][step]
            xs = np.arange(len(vals)) + ci * width
            ax.bar(xs, vals, width=width, label=case,
                   color=case_color(ci), alpha=0.7)
        ax.set_xlabel("Action dim")
        ax.set_ylabel("RMS displacement")
        ax.set_title(f"{step * 4 / 1e6:.1f}M steps")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("naction — per-dimension RMS displacement", fontsize=14)
    plt.tight_layout()
    save_fig(fig, out_path)


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    required=True)
    parser.add_argument("--cases",       required=True, nargs="+")
    parser.add_argument("--scales",      nargs="+", type=float, default=None,
                        help="Scale per case (positionally matched to --cases). "
                             "Required for perturbation-to-prior ratio.")
    parser.add_argument("--steps",       nargs="+", type=int,
                        default=[100000, 200000, 400000, 600000])
    parser.add_argument("--output-dir",  default=None,
                        help="Default: <data-dir>/metrics")
    parser.add_argument("--knn-k",       type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.output_dir or os.path.join(args.data_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)

    data = load_all(args.data_dir, args.cases, args.steps)
    case_names = args.cases
    steps = args.steps

    # build case→scale mapping
    scale_map = {}
    if args.scales is not None:
        assert len(args.scales) == len(args.cases), \
            f"--scales ({len(args.scales)}) must match --cases ({len(args.cases)})"
        scale_map = dict(zip(args.cases, args.scales))

    # valid steps per case (only those that were loaded)
    valid_steps = {
        case: sorted(s for s in steps if s in data[case])
        for case in case_names
    }
    all_valid_steps = sorted(set(s for vs in valid_steps.values() for s in vs))

    for key in METRIC_KEYS:
        print(f"\n{'='*60}")
        print(f"  Computing metrics for: {key}")
        print(f"{'='*60}")

        base_arr = data["base"][key]  # (N, D)

        # ---- 1. L2 displacement ----
        l2_dict = {}
        for case in case_names:
            l2_dict[case] = {}
            for step in valid_steps[case]:
                d = l2_displacement(base_arr, data[case][step][key])
                l2_dict[case][step] = (d.mean(), d.std())
                print(f"  L2  {case}/step={step}: {d.mean():.4f} ± {d.std():.4f}")

        plot_scalar_vs_step(l2_dict, all_valid_steps, case_names,
                            r"$\tilde{z}$ L2 displacement", f"{key} — L2 displacement",
                            os.path.join(out_dir, f"{key}_l2.pdf"))

        # ---- 2. Cosine similarity ----
        cos_dict = {}
        for case in case_names:
            cos_dict[case] = {}
            for step in valid_steps[case]:
                c = cosine_similarity(base_arr, data[case][step][key])
                cos_dict[case][step] = (c.mean(), c.std())
                print(f"  cos {case}/step={step}: {c.mean():.4f} ± {c.std():.4f}")

        plot_scalar_vs_step(cos_dict, all_valid_steps, case_names,
                            "Cosine similarity", f"{key} — cosine similarity",
                            os.path.join(out_dir, f"{key}_cosine.pdf"))

        # ---- 3. k-NN preservation ----
        knn_dict = {}
        for case in case_names:
            knn_dict[case] = {}
            for step in valid_steps[case]:
                ov = knn_preservation(base_arr, data[case][step][key], k=args.knn_k)
                knn_dict[case][step] = (ov.mean(), ov.std())
                print(f"  kNN {case}/step={step}: {ov.mean():.4f} ± {ov.std():.4f}")

        plot_scalar_vs_step(knn_dict, all_valid_steps, case_names,
                            f"k-NN preservation (k={args.knn_k})",
                            f"{key} — k-NN preservation",
                            os.path.join(out_dir, f"{key}_knn.pdf"))

        # ---- 4. Per-dimension RMS (only for naction) ----
        if key == "naction":
            dim_dict = {}
            for case in case_names:
                dim_dict[case] = {}
                for step in valid_steps[case]:
                    rms = per_dim_rms(base_arr, data[case][step][key])
                    dim_dict[case][step] = rms

            plot_per_dim_action(dim_dict, all_valid_steps, case_names,
                                os.path.join(out_dir, f"naction_per_dim.pdf"))

        # ---- 5. Perturbation-to-prior ratio (only for modified_obs_emb) ----
        if key == "modified_obs_emb" and scale_map:
            z_logvar_base = data["base"].get("z_logvar")
            if z_logvar_base is None:
                print("  [WARN] base_recon.pt missing z_logvar, skipping perturbation ratio")
            else:
                ptr_dict = {}
                for case in case_names:
                    ptr_dict[case] = {}
                    scale = scale_map[case]
                    for step in valid_steps[case]:
                        res_z = data[case][step].get("res_z")
                        if res_z is None:
                            print(f"  [WARN] {case}/step={step} missing res_z, skipping")
                            continue
                        r = perturbation_to_prior_ratio(z_logvar_base, res_z, scale)
                        ptr_dict[case][step] = (r.mean(), r.std())
                        print(f"  P/P  {case}/step={step} (scale={scale}): "
                              f"{r.mean():.4f} ± {r.std():.4f}")

                plot_scalar_vs_step(
                    ptr_dict, all_valid_steps, case_names,
                    "Perturbation / Prior ratio",
                    "modified_obs_emb — perturbation-to-prior ratio",
                    os.path.join(out_dir, "modified_obs_emb_perturbation_ratio.pdf"))

        # ---- 6. Mahalanobis OOD score (only for modified_obs_emb) ----
        #    Use median + IQR instead of mean ± std (right-skewed distribution)
        if key == "modified_obs_emb":
            print("  Fitting Mahalanobis (Ledoit-Wolf) on base ...")
            maha_dict = {}   # case -> step -> (median, q1, q3)
            for case in case_names:
                maha_dict[case] = {}
                for step in valid_steps[case]:
                    m = mahalanobis_ood(base_arr, data[case][step][key])
                    q1, median, q3 = np.percentile(m, [25, 50, 75])
                    maha_dict[case][step] = (median, q1, q3)
                    print(f"  Maha {case}/step={step}: "
                          f"median={median:.1f}  IQR=[{q1:.1f}, {q3:.1f}]")

            plot_median_iqr_vs_step(
                maha_dict, all_valid_steps, case_names,
                r"$\tilde{c}$ Mahalanobis dist.",
                "modified_obs_emb — Mahalanobis OOD",
                os.path.join(out_dir, "modified_obs_emb_mahalanobis.pdf"))

    # ---- 7. Mahalanobis of perturbed_z w.r.t. base z_mean distribution ----
    if scale_map:
        z_mean_base = data["base"].get("z_mean")
        if z_mean_base is None:
            print("  [WARN] base_recon.pt missing z_mean, skipping z-space Mahalanobis")
        else:
            print(f"\n{'='*60}")
            print(f"  Computing z-space Mahalanobis (perturbed_z vs base z_mean)")
            print(f"{'='*60}")
            maha_z_dict = {}
            for case in case_names:
                maha_z_dict[case] = {}
                scale = scale_map[case]
                for step in valid_steps[case]:
                    z_mean_case = data[case][step].get("z_mean")
                    res_z = data[case][step].get("res_z")
                    if z_mean_case is None or res_z is None:
                        print(f"  [WARN] {case}/step={step} missing z_mean or res_z, skipping")
                        continue
                    perturbed_z = z_mean_case + scale * res_z
                    m = mahalanobis_ood(z_mean_base, perturbed_z)
                    q1, median, q3 = np.percentile(m, [25, 50, 75])
                    maha_z_dict[case][step] = (median, q1, q3)
                    print(f"  Maha_z {case}/step={step} (scale={scale}): "
                          f"median={median:.1f}  IQR=[{q1:.1f}, {q3:.1f}]")

            plot_median_iqr_vs_step(
                maha_z_dict, all_valid_steps, case_names,
                r"$\tilde{z}$ Mahalanobis dist.",
                r"$z$-space Mahalanobis OOD ($\tilde{z}$ vs $z$)",
                os.path.join(out_dir, "z_mahalanobis.pdf"))

    print(f"\nAll figures saved to {out_dir}")
