import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FormatStrFormatter


INPUT_DIR = Path("data/plot/action_compare_rollouts")
OUTPUT_PATH = Path("data/plot/action_compare_rollouts/action_compare_rollouts_3x2.pdf")
ARIAL_FONT_PATH = Path("data/Arial.ttf")
CAMBRIA_MATH_FONT_PATH = Path("data/cambria-math.ttf")
STIX_MATH_FONT_PATH = Path("data/STIXTwoMath.otf")

ROLLOUTS = [
    (100016, "x"),
    (100088, "y"),
    (100081, "z"),
]
METHODS = ["podec", "zprl"]
PALETTE = {
    "podec": "#696fa2",
    "zprl": "#c66a42",
    "base": "#000000",
}
TITLE_MAP = {
    "podec": "Po-Dec",
    "zprl": "ZPRL",
}
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


for font_path in [ARIAL_FONT_PATH, CAMBRIA_MATH_FONT_PATH, STIX_MATH_FONT_PATH]:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))

if ARIAL_FONT_PATH.exists():
    mpl.rcParams["font.family"] = font_manager.FontProperties(
        fname=str(ARIAL_FONT_PATH)
    ).get_name()
else:
    mpl.rcParams["font.family"] = [
        "Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"]

math_font_name = "Cambria Math"
if CAMBRIA_MATH_FONT_PATH.exists():
    math_font_name = font_manager.FontProperties(
        fname=str(CAMBRIA_MATH_FONT_PATH)
    ).get_name()

mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = math_font_name
mpl.rcParams["mathtext.it"] = math_font_name
mpl.rcParams["mathtext.bf"] = math_font_name
mpl.rcParams["mathtext.cal"] = math_font_name
mpl.rcParams["mathtext.sf"] = math_font_name
mpl.rcParams["mathtext.fallback"] = "stix"


def load_rollout(input_dir, method, seed, axis):
    path = input_dir / f"{method}_seed={seed}_{axis}.npz"
    with np.load(path) as payload:
        return (
            payload["time_seconds"].copy(),
            payload["base_action"].copy(),
            payload["sum_action"].copy(),
        )


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#ddd7cb", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b6256")
    ax.spines["bottom"].set_color("#6b6256")
    ax.tick_params(colors="#4c463d", labelsize=11)


def plot_grid(input_dir, output_path):
    fig, axes = plt.subplots(
        3, 2, figsize=(9, 7.0), dpi=220,
        sharex=False, sharey="row")

    panel_idx = 0
    for row, (seed, axis) in enumerate(ROLLOUTS):
        row_data = {
            method: load_rollout(input_dir, method, seed, axis)
            for method in METHODS
        }
        values = np.concatenate([
            values
            for method in METHODS
            for values in row_data[method][1:]
        ])
        y_min = float(np.min(values))
        y_max = float(np.max(values))
        if np.isclose(y_min, y_max):
            delta = 1e-3 if np.isclose(y_min, 0.0) else abs(y_min) * 0.05
            y_min -= delta
            y_max += delta
        y_ticks = np.linspace(y_min, y_max, 5)

        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            time, base_action, sum_action = row_data[method]
            style_axis(ax)
            ax.plot(
                time, base_action,
                color=PALETTE["base"], linewidth=2.0,
                linestyle=(0, (5, 3)))
            ax.plot(
                time, sum_action,
                color=PALETTE[method], linewidth=2.6)
            ax.set_title(
                f"{PANEL_LABELS[panel_idx]} {TITLE_MAP[method]} - Seed={seed}",
                fontsize=12, pad=7)
            if row == len(ROLLOUTS) - 1:
                ax.set_xlabel("Time (s)", fontsize=11)
            ax.set_ylabel(f"{axis.upper()} position", fontsize=11)
            ax.set_ylim(y_min, y_max)
            ax.yaxis.set_major_locator(FixedLocator(y_ticks))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            panel_idx += 1

    handles = [
        plt.Line2D(
            [], [], color=PALETTE["base"], linewidth=2.0,
            linestyle=(0, (5, 3)), label="Base action"),
        plt.Line2D(
            [], [], color=PALETTE["podec"], linewidth=2.6,
            label="Po-Dec action"),
        plt.Line2D(
            [], [], color=PALETTE["zprl"], linewidth=2.6,
            label="ZPRL action"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.005), fontsize=11)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    plot_grid(args.input_dir, args.output)


if __name__ == "__main__":
    main()
