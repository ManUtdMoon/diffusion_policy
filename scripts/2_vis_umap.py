"""
Visualise UMAP projections from umap.pkl.

For each embedding key (e.g. modified_obs_emb, naction) and each step,
one figure is produced with one subplot per case (base grey + case colour).

Usage:
  python scripts/2_vis_umap.py \\
      --umap-pkl  data/umap/umap.pkl \\
      --scales    0.1 0.2 0.5 \\
      [--output-dir data/umap/figs] \\
      [--dim 2] \\
      [--alpha 0.5] \\
      [--point-size 5]
"""
import argparse
import os
import pickle
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

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

# ------------------------------------------------------------------ #
# colours
# ------------------------------------------------------------------ #
BASE_COLOR  = "#aaaaaa"
CASE_COLORS = [
    "#e41a1c", "#ff7f00", "#4daf4a", "#984ea3",
    "#377eb8", "#a65628", "#f781bf", "#999999",
]

CASE_COLORS = [
    '#e8cd81', '#c66a42', '#6a4a2b',
]

# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #

def scatter2d(ax, xy, color, label, alpha, s):
    ax.scatter(xy[:, 0], xy[:, 1], c=color, s=s, alpha=alpha,
               linewidths=0, rasterized=True, label=label)


def make_legend(ax, case_label, case_color):
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=BASE_COLOR, markersize=6, label="base"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=case_color, markersize=6, label=case_label),
    ]
    ax.legend(handles=handles, loc="best", fontsize=11, framealpha=0.7)


def scale_label(scale_val):
    """Format scale value as a LaTeX lambda label."""
    return rf"$\lambda={scale_val}$"


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--umap-pkl",    required=True)
    parser.add_argument("--scales",      required=True, nargs="+", type=float,
                        help="Scale values, positionally matched to cases in the pkl")
    parser.add_argument("--output-dir",  default=None,
                        help="Default: same directory as umap.pkl")
    parser.add_argument("--dim",         type=int, default=2, choices=[2, 3])
    parser.add_argument("--alpha",       type=float, default=0.5)
    parser.add_argument("--point-size",  type=float, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(args.umap_pkl)

    with open(args.umap_pkl, "rb") as f:
        umap_data = pickle.load(f)

    # select keys matching requested dim
    dim_suffix = f"_{args.dim}d"
    keys = [k for k in umap_data if k.endswith(dim_suffix)]
    if not keys:
        raise ValueError(f"No keys ending with '{dim_suffix}' found in {args.umap_pkl}. "
                         f"Available: {list(umap_data.keys())}")

    for emb_key in keys:
        proj = umap_data[emb_key]       # {"base": (N,2), "case": {step: (N,2)}, ...}
        base_xy = proj["base"]          # (N, 2)

        # collect cases (everything except "base")
        case_names = [k for k in proj if k != "base"]

        # build case→scale and case→label mappings
        assert len(args.scales) == len(case_names), \
            f"--scales ({len(args.scales)}) must match number of cases ({len(case_names)}): {case_names}"
        case_scale = dict(zip(case_names, args.scales))
        case_label = {name: scale_label(s) for name, s in case_scale.items()}
        case_color = {name: CASE_COLORS[i % len(CASE_COLORS)]
                      for i, name in enumerate(case_names)}

        # collect all steps across cases
        all_steps = set()
        for name in case_names:
            all_steps.update(proj[name].keys())
        steps = sorted(all_steps)

        fig_dir = os.path.join(output_dir, emb_key)
        os.makedirs(fig_dir, exist_ok=True)

        for step in steps:
            active_cases = [n for n in case_names if step in proj[n]]
            ncols = len(active_cases)
            fig, axes = plt.subplots(1, ncols,
                                     figsize=(4 * ncols, 4),
                                     squeeze=False)
            axes = axes[0]

            for ci, name in enumerate(active_cases):
                ax = axes[ci]
                color = case_color[name]
                label = case_label[name]
                scatter2d(ax, base_xy, BASE_COLOR, "base",
                          args.alpha * 0.4, args.point_size * 0.8)
                scatter2d(ax, proj[name][step], color, label,
                          args.alpha, args.point_size)
                ax.set_title(label, fontsize=14)
                ax.set_aspect("equal", "box")
                make_legend(ax, label, color)

            plt.tight_layout()
            out_path = os.path.join(fig_dir, f"step={step}.pdf")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out_path}")

    print("Done.")
