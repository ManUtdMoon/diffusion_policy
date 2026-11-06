import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from zprl.common.action_mse_util import U


def read_candidates(directory):
    path = directory / "quantiles.csv"
    fields = ["epoch", *(f"q{round(100 * u):02d}" for u in U)]
    rows = []
    with path.open(newline="") as file:
        reader = csv.reader(file)
        if next(reader, None) != fields:
            raise ValueError(f"Missing or invalid header in {path}")
        for row in reader:
            if len(row) != len(fields):
                raise ValueError(f"Expected {len(fields)} columns at {path}:{reader.line_num}")
            epoch = int(row[0])
            checkpoint = directory / f"epoch_{epoch:04d}.ckpt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing candidate {checkpoint}")
            rows.append((epoch, [float(value) for value in row[1:]]))
    if not rows:
        raise ValueError(f"No candidates in {path}")
    rows.sort(key=lambda row: row[0])
    epochs, qbar = zip(*rows)
    if len(set(epochs)) != len(epochs):
        raise ValueError(f"Duplicate candidate epoch in {path}")
    qbar = np.asarray(qbar, dtype=np.float64)
    if not np.isfinite(qbar).all():
        raise ValueError(f"Non-finite quantile in {path}")
    return np.asarray(epochs), qbar


def compute_risk(qbar):
    log_mse = np.log(np.maximum(qbar, np.finfo(np.float32).tiny))
    span = np.ptp(log_mse, axis=0)
    normalized = np.divide(log_mse - log_mse.min(axis=0), span,
                           out=np.zeros_like(log_mse), where=span > 0)
    low = np.asarray(U) < 0.50
    coverage = normalized[:, low].mean(axis=1)
    reliability = normalized[:, ~low].mean(axis=1)
    return coverage, reliability, np.maximum(coverage, reliability)


def save_plots(directory, epochs, qbar, coverage, reliability, risk, best):
    selected = epochs[best]
    fig, ax = plt.subplots(layout="constrained")
    cmap = LinearSegmentedColormap.from_list(
        "quantile", ["#c96b66", "#df9d9a", "#c9c4c6", "#8aa8c8", "#4a6f96"])
    colors = plt.cm.ScalarMappable(norm=plt.Normalize(U[0], U[-1]), cmap=cmap)
    for u, values in zip(U[::2], qbar[:, ::2].T):
        ax.plot(epochs, values, marker=".", color=colors.to_rgba(u))
    ax.axvline(selected, color="black", linestyle=":", label=f"Selected epoch {selected}")
    ax.set(xlabel="Candidate epoch", ylabel="Mean validation action MSE quantile")
    ax.legend()
    fig.colorbar(colors, ax=ax, orientation="horizontal", shrink=0.6,
                 ticks=[U[0], 0.5, U[-1]], format="%.2f", label="Quantile $u$")
    fig.savefig(directory / "quantiles.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(epochs, coverage, label="coverage")
    ax.plot(epochs, reliability, label="reliability")
    ax.plot(epochs, risk, linestyle="--", label="cov_rel_risk")
    ax.scatter(selected, risk[best], color="black", zorder=3, clip_on=False,
               label=f"Selected epoch {selected}")
    ax.set(xlabel="Candidate epoch", ylabel="Normalized risk", ylim=(0, 1))
    ax.legend()
    fig.savefig(directory / "normalized_risk.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Select a checkpoint by cov_rel_risk.")
    parser.add_argument("run_dir", type=Path)
    directory = parser.parse_args().run_dir / "checkpoints"
    try:
        epochs, qbar = read_candidates(directory)
        coverage, reliability, risk = compute_risk(qbar)
        best = risk.argmin()
        save_plots(directory, epochs, qbar, coverage, reliability, risk, best)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"epoch_{epochs[best]:04d}.ckpt")


if __name__ == "__main__":
    main()
