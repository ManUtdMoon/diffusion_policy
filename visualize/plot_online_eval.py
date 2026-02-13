#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot eval.txt (chunk_step, success_rate, episode_length) for online runs."
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        required=True,
        help="Path to eval.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Default: <eval dir>/online_eval.png",
    )
    parser.add_argument(
        "--chunk-ratio",
        type=float,
        default=16.0,
        help="Env step conversion ratio for chunk step (default: 16).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Moving-average window for filtering (set 1 to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eval_path = args.eval_path.expanduser().resolve()
    if not eval_path.is_file():
        raise FileNotFoundError(f"eval.txt not found: {eval_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else eval_path.parent / f"online_eval_sm{args.smooth_window}.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # File format: chunk_step success_rate episode_length
    df = pd.read_csv(
        eval_path,
        sep=r"\s+",
        header=None,
        names=["chunk_step", "success_rate", "episode_length"],
    )
    if df.empty:
        raise ValueError(f"No rows found in {eval_path}")

    for col in ["chunk_step", "success_rate", "episode_length"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["chunk_step", "success_rate", "episode_length"]).copy()
    if df.empty:
        raise ValueError(f"No valid numeric rows found in {eval_path}")

    df["env_steps_k"] = df["chunk_step"] * float(args.chunk_ratio) / 1000.0
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be a positive integer.")

    success_smooth = df["success_rate"].rolling(
        window=args.smooth_window, min_periods=1, center=True
    ).mean()
    episode_len_smooth = df["episode_length"].rolling(
        window=args.smooth_window, min_periods=1, center=True
    ).mean()

    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax2 = ax1.twinx()

    # ax1.plot(
    #     df["env_steps_k"],
    #     df["success_rate"],
    #     color="#1f77b4",
    #     linewidth=1.0,
    #     alpha=0.25,
    #     marker="o",
    #     label="success_rate_raw",
    # )
    ax1.plot(
        df["env_steps_k"],
        success_smooth,
        color="#1f77b4",
        linewidth=1.8,
        marker="o",
        label=f"SR",
    )
    # ax2.plot(
    #     df["env_steps_k"],
    #     df["episode_length"],
    #     color="#d62728",
    #     linewidth=1.0,
    #     alpha=0.25,
    #     marker="o",
    #     label="episode_length_raw",
    # )
    ax2.plot(
        df["env_steps_k"],
        episode_len_smooth,
        color="#d62728",
        linewidth=1.8,
        marker="o",
        label=f"Epi. Len.",
    )

    ax1.set_ylim(0.5, 1.0)
    ax1.set_xlabel(r"Env steps ($\times 10^3$)")
    ax1.set_ylabel("Success Rate", color="#1f77b4")
    ax2.set_ylim(100, 125)
    ax2.set_ylabel("Episode Length", color="#d62728")
    ax1.set_title(f"FlipEgg")
    ax1.grid(True, alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
