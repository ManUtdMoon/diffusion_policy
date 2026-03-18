import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FixedLocator


PLOT_DIR = Path("data/plot/action")
ARIAL_FONT_PATH = Path("data/Arial.ttf")
CAMBRIA_MATH_FONT_PATH = Path("data/cambria-math.ttf")
STIX_MATH_FONT_PATH = Path("data/STIXTwoMath.otf")

PALETTE = {
    "zprl": "#c66a42",
    "resrl": "#696fa2",
    "base": "#000000",
}
TITLE_MAP = {
    "zprl": "ZPRL",
    "resrl": "Po-Dec",
}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]
STEP_RATIO = 8


for font_path in [ARIAL_FONT_PATH, CAMBRIA_MATH_FONT_PATH, STIX_MATH_FONT_PATH]:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))

if ARIAL_FONT_PATH.exists():
    mpl.rcParams["font.family"] = font_manager.FontProperties(
        fname=str(ARIAL_FONT_PATH)
    ).get_name()
else:
    mpl.rcParams["font.family"] = ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"]

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


def load_pair(mode, step):
    payload = np.load(PLOT_DIR / f"{mode}_step={step}.npz")
    return payload["action_seq"], payload["base_action_seq"]


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#ddd7cb", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b6256")
    ax.spines["bottom"].set_color("#6b6256")
    ax.tick_params(colors="#4c463d", labelsize=11)


def format_step_label(step):
    scaled_step = step * STEP_RATIO
    return f"{scaled_step / 1e6:.1f}M"


def format_panel_title(panel_idx, mode, step):
    return f"{PANEL_LABELS[panel_idx]} {TITLE_MAP[mode]} @ Steps={format_step_label(step)}"


def get_time_axis(length):
    return np.arange(length) / 20.0


def compute_shared_yticks(cases, axis_name):
    axis_idx = AXIS_INDEX[axis_name]
    values = []
    for mode, step in cases:
        action_seq, base_action_seq = load_pair(mode, step)
        values.append(action_seq[0, :, axis_idx])
        values.append(base_action_seq[0, :, axis_idx])

    y_all = np.concatenate(values)
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    if np.isclose(y_min, y_max):
        delta = 1e-3 if np.isclose(y_min, 0.0) else abs(y_min) * 0.05
        y_min -= delta
        y_max += delta

    ticks = np.linspace(y_min, y_max, 5)
    return y_min, y_max, ticks


def plot_grid(axis_name, step, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 4.8), dpi=220, sharex=False, sharey=True)
    cases = [
        ("resrl", 0),
        ("resrl", step),
        ("zprl", 0),
        ("zprl", step),
    ]
    axis_idx = AXIS_INDEX[axis_name]
    y_min, y_max, y_ticks = compute_shared_yticks(cases, axis_name)

    for idx, (ax, (mode, current_step)) in enumerate(zip(axes.flat, cases)):
        action_seq, base_action_seq = load_pair(mode, current_step)
        action = action_seq[0, :, axis_idx]
        base = base_action_seq[0, :, axis_idx]
        t = get_time_axis(action.shape[0])

        style_axis(ax)
        ax.plot(t, base, color=PALETTE["base"], linewidth=2.0, linestyle=(0, (5, 3)))
        ax.plot(t, action, color=PALETTE[mode], linewidth=2.6)
        ax.set_title(format_panel_title(idx, mode, current_step), fontsize=12, pad=7)
        if idx >= 2:
            ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel(f"{axis_name.upper()} position", fontsize=11)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    handles = [
        plt.Line2D([], [], color=PALETTE["base"], linewidth=2.0, linestyle=(0, (5, 3)), label="Base action"),
        plt.Line2D([], [], color=PALETTE["resrl"], linewidth=2.6, label="Po-Dec action"),
        plt.Line2D([], [], color=PALETTE["zprl"], linewidth=2.6, label="ZPRL action"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = output_dir / f"compare_grid_step={step}_{axis_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=["x", "y", "z"], default="x")
    parser.add_argument("--step", type=int, default=100_000)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = PLOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_grid(args.axis, args.step, output_dir)


if __name__ == "__main__":
    main()
