#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

METHOD_COLORS = {
    "ZPRL": "#E63983",
    "Po-dec": "#56B4E9",
}
METHOD_MARKERS = {
    "ZPRL": "o",
    "Po-dec": "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot comparison curves from eval-*.txt under one directory."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Directory containing eval-*.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: --eval-dir",
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
        default=1,
        help="Moving-average window for filtering (set 1 to disable).",
    )
    return parser.parse_args()


def canonical_method_name(raw_name: str) -> str:
    name = raw_name.strip().lower()
    if name == "zprl":
        return "ZPRL"
    if name in {"po-dec", "podec"}:
        return "Po-dec"
    return raw_name


def load_eval_file(eval_path: Path, chunk_ratio: float, smooth_window: int) -> pd.DataFrame:
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

    df = df.sort_values("chunk_step").reset_index(drop=True)
    df["env_steps_k"] = df["chunk_step"] * float(chunk_ratio) / 1000.0

    df["success_rate"] = df["success_rate"].rolling(
        window=smooth_window, min_periods=1, center=True
    ).mean()
    df["episode_length"] = df["episode_length"].rolling(
        window=smooth_window, min_periods=1, center=True
    ).mean()
    return df


def main() -> None:
    args = parse_args()
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be a positive integer.")

    eval_dir = args.eval_dir.expanduser().resolve()
    if not eval_dir.is_dir():
        raise NotADirectoryError(f"Directory not found: {eval_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else eval_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    TITLE = None
    if "flip" in eval_dir.name.lower():
        TITLE = "Flip Egg"
    elif "juicing" in eval_dir.name.lower():
        TITLE = "Place Orange"
    elif "box" in eval_dir.name.lower():
        TITLE = "Open Box"
    else:
        raise ValueError("Unrecognized task. Only accept 'flip', 'juicing', or 'box'.")

    eval_files = sorted(eval_dir.glob("eval-*.txt"))
    if not eval_files:
        raise FileNotFoundError(f"No eval-*.txt found under: {eval_dir}")

    method_to_df: Dict[str, pd.DataFrame] = {}
    for eval_file in eval_files:
        raw_method = eval_file.stem[len("eval-") :]
        method = canonical_method_name(raw_method)
        method_to_df[method] = load_eval_file(
            eval_file,
            chunk_ratio=args.chunk_ratio,
            smooth_window=args.smooth_window,
        )

    # 1) SR figure
    fig, ax = plt.subplots(figsize=(5, 4))
    for method, df in method_to_df.items():
        color = METHOD_COLORS.get(method, None)
        marker = METHOD_MARKERS.get(method, "o")
        ax.plot(
            df["env_steps_k"],
            df["success_rate"],
            linewidth=2.0,
            marker=marker,
            label=method,
            color=color,
        )
    ax.set_xlabel(r"Env steps ($\times 10^3$)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(top=1.0)
    ax.set_title(TITLE)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    sr_out = output_dir / f"online_eval_sr_sm{args.smooth_window}.pdf"
    fig.savefig(sr_out, dpi=200)
    plt.close(fig)
    print(f"Saved: {sr_out}")

    # 2) Episode length figure
    fig, ax = plt.subplots(figsize=(5, 4))
    for method, df in method_to_df.items():
        color = METHOD_COLORS.get(method, None)
        marker = METHOD_MARKERS.get(method, "o")
        ax.plot(
            df["env_steps_k"],
            df["episode_length"],
            linewidth=2.0,
            marker=marker,
            label=method,
            color=color,
        )
    ax.set_xlabel(r"Env steps ($\times 10^3$)")
    ax.set_ylabel("Episode Length")
    ax.set_title(TITLE)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    epi_out = output_dir / f"online_eval_epi_len_sm{args.smooth_window}.pdf"
    fig.savefig(epi_out, dpi=200)
    plt.close(fig)
    print(f"Saved: {epi_out}")


if __name__ == "__main__":
    main()
