import click
import numpy as np
import pandas as pd
from pathlib import Path


THRESHOLD = 0.8
TAIL_POINTS = 5
BASE_DIR = Path("data/plot")


def load_run_stats(run_dir: Path):
    train_dir = run_dir / "train"
    sr_path = train_dir / "sr.csv"
    interval_path = train_dir / "interval.txt"
    start_path = train_dir / "start.txt"
    ta_path = train_dir / "Ta.txt"

    if not sr_path.is_file() or not interval_path.is_file():
        raise FileNotFoundError(f"Missing required train files in {train_dir}")

    sr_values = pd.read_csv(sr_path, header=None)[0].to_numpy(dtype=float)
    if len(sr_values) == 0:
        raise ValueError(f"Empty sr.csv in {train_dir}")

    interval = int(interval_path.read_text().strip())
    start = int(start_path.read_text().strip()) if start_path.is_file() else 0
    ta = float(ta_path.read_text().strip()) if ta_path.is_file() else 1.0

    crossing_indices = np.flatnonzero(sr_values > THRESHOLD)
    if len(crossing_indices) == 0:
        steps_to_threshold = np.nan
    else:
        first_idx = int(crossing_indices[0])
        steps_to_threshold = (start + first_idx * interval) * ta

    final_sr = float(np.mean(sr_values[-TAIL_POINTS:]))
    return steps_to_threshold, final_sr


def summarize_algo(task_dir: Path, algo_dir: Path):
    run_dirs = sorted(
        d for d in algo_dir.iterdir() if d.is_dir() and d.name.startswith("run")
    )
    if not run_dirs:
        return None

    steps_to_threshold_values = []
    final_sr_values = []
    missing_threshold_runs = []

    for run_dir in run_dirs:
        steps_to_threshold, final_sr = load_run_stats(run_dir)
        if np.isnan(steps_to_threshold):
            missing_threshold_runs.append(run_dir.name)
        else:
            steps_to_threshold_values.append(steps_to_threshold)
        final_sr_values.append(final_sr)

    if not final_sr_values:
        return None

    return {
        "task": task_dir.name,
        "algorithm": algo_dir.name,
        "num_runs": len(run_dirs),
        "num_threshold_runs": len(steps_to_threshold_values),
        "steps_to_0.8": np.mean(steps_to_threshold_values) / 1e6 if steps_to_threshold_values else np.nan,
        "final_sr": np.mean(final_sr_values),
        "missing_threshold_runs": ",".join(missing_threshold_runs) if missing_threshold_runs else "",
    }


@click.command()
@click.option('--task', 'tasks', required=True, multiple=True, help='Task(s) to summarize, e.g. --task can --task square')
def main(tasks):
    """Summarize train stats from processed plot data for one or more tasks."""
    rows = []

    for task in tasks:
        task_dir = BASE_DIR / task
        if not task_dir.is_dir():
            raise click.BadParameter(f"Task directory not found: {task_dir}")

        algo_dirs = sorted(d for d in task_dir.iterdir() if d.is_dir())
        if not algo_dirs:
            click.echo(f"Warning: no algorithm directories found in {task_dir}")
            continue

        for algo_dir in algo_dirs:
            summary = summarize_algo(task_dir, algo_dir)
            if summary is None:
                click.echo(f"Warning: skipped {algo_dir} because no valid run directories were found")
                continue
            rows.append(summary)

    if not rows:
        click.echo("No valid statistics found.")
        return

    df = pd.DataFrame(rows)
    df["steps_to_0.8"] = df["steps_to_0.8"].map(
        lambda x: "N/A" if pd.isna(x) else f"{x:.3f}"
    )
    df["final_sr"] = df["final_sr"].map(lambda x: f"{x:.3f}")

    click.echo(df.to_string(index=False))


if __name__ == '__main__':
    main()
