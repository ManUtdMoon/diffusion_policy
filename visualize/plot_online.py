#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Edit this section to choose which metrics to visualize.
SUCCESS_KEY = "info/recent_done_sr"
EPISODE_LEN_KEY = "info/recent_done_avg_len"
OTHER_METRICS = [
    # "info/rewards",
    # "info/actor_entropy",
    # "info/q_predicted",
    # "loss/critic_loss",
    # "loss/actor_loss",
]
SMOOTH_WINDOW = 151


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot online training metrics from a JSONL log file."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
        help="Path to merged logs.json.txt (JSON Lines).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        required=True,
        help="Keep rows where info/global_step % interval == 0.",
    )
    parser.add_argument(
        "--step-ratio",
        type=float,
        required=True,
        help="x-axis scale ratio: Env steps = global_step * step_ratio / 1k.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save figures. Default: <log file dir>/plots_online",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be a positive integer.")

    log_path = args.log_path.expanduser().resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else log_path.parent / "plots_online"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    task = args.task if args.task is not None else log_path.stem

    df = pd.read_json(log_path, lines=True)
    if "info/global_step" not in df.columns:
        raise KeyError('Missing required column: "info/global_step"')

    # Keep only rows with valid integer global_step.
    df = df.dropna(subset=["info/global_step"]).copy()
    df["info/global_step"] = pd.to_numeric(df["info/global_step"], errors="coerce")
    df = df.dropna(subset=["info/global_step"]).copy()
    df["info/global_step"] = df["info/global_step"].astype("int64")

    # Apply moving-average smoothing before downsampling.
    metrics_to_smooth = [SUCCESS_KEY, EPISODE_LEN_KEY] + OTHER_METRICS
    for metric in metrics_to_smooth:
        if metric not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors="coerce")
        df[metric] = series.rolling(
            window=SMOOTH_WINDOW,
            min_periods=1,
            center=True,
        ).mean()

    # Downsample by step modulo to handle missing/uniformly logged steps.
    df = df[df["info/global_step"] % args.interval == 0].copy()
    if df.empty:
        raise ValueError("No data left after modulo filtering. Check --interval.")

    # Convert to Env steps in thousands (1k).
    df["env_steps_k"] = df["info/global_step"] * float(args.step_ratio) / 1000.0

    # Check monotonic increase for plotting sanity.
    if not df["info/global_step"].is_monotonic_increasing:
        print("Warning: info/global_step is not monotonic increasing after downsampling.")

    # 1) Success rate + episode length in one figure.
    if SUCCESS_KEY in df.columns or EPISODE_LEN_KEY in df.columns:
        fig, ax1 = plt.subplots(figsize=(5, 4))
        ax2 = ax1.twinx()

        has_any = False
        if SUCCESS_KEY in df.columns:
            y1 = pd.to_numeric(df[SUCCESS_KEY], errors="coerce")
            ax1.plot(df["env_steps_k"], y1, color="#1f77b4", linewidth=1.5, label="success_rate")
            has_any = True
        if EPISODE_LEN_KEY in df.columns:
            y2 = pd.to_numeric(df[EPISODE_LEN_KEY], errors="coerce")
            ax2.plot(df["env_steps_k"], y2, color="#d62728", linewidth=1.5, label="episode_length")
            has_any = True

        if has_any:
            ax1.set_xlabel(r"Env steps ($\times 10^3$)")
            ax1.set_ylabel("Success Rate (%)", color="#1f77b4")
            ax2.set_ylabel("Episode Length", color="#d62728")
            ax1.grid(True, alpha=0.3)
            ax1.set_title(task)

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

            fig.tight_layout()
            out_path = output_dir / "success_and_episode_length.pdf"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"Saved: {out_path}")

    # 2) Other metrics, one figure per metric.
    for metric in OTHER_METRICS:
        if metric not in df.columns:
            print(f"Skip missing metric: {metric}")
            continue

        y = pd.to_numeric(df[metric], errors="coerce")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["env_steps_k"], y, linewidth=1.5)
        ax.set_xlabel(r"Env steps ($\times 10^3$)")
        ax.set_ylabel(metric)
        ax.set_title(task)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_name = metric.replace("/", "_")
        out_path = output_dir / f"{safe_name}.pdf"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
