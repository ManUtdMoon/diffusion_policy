import click
import numpy as np
import pandas as pd
from pathlib import Path


THRESHOLDS = (0.9, 0.95)
TAIL_POINTS = 5
SMOOTH_WINDOW = 3
PLOT_ROOT = Path(__file__).resolve().parent.parent / "data" / "plot"
TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "cfg_ref" / "train_stats.csv"
TABLE_ALGORITHMS = ("DSRL", "Po-dec", "ZPRL")
TABLE_ALGO_NAMES = {"Po-dec": "Po-Dec"}
END_STEP_BY_TASK = {
    "can": 1e6,
    "square": 3e6,
    "transport": 3e6,

    "door": 2e6,
    "hammer": 1e6,
    "pen": 2e6,

    "metaworld_box-close": 1e6,
    "metaworld_push-wall": 1e6,

    "num_demo": 3e6,
    "dim": 3e6,
    "scale": 3e6,
}


def resolve_task_dir(task: str) -> Path:
    candidates = [
        PLOT_ROOT / task,
        PLOT_ROOT / 'sim-main' / task,
        PLOT_ROOT / "done" / "exp-1-main" / task,
        PLOT_ROOT / "done" / f"exp-2-{task}",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise click.BadParameter(
        f"Task directory not found for '{task}'. Checked: {', '.join(str(path) for path in candidates)}"
    )


def smooth(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def rising_start_index(values: np.ndarray) -> int:
    """Index where the final rise towards the peak begins.

    Some curves (e.g. `can`) dip before climbing, so the very first crossing of a
    threshold can happen on the way *down* and is not what we want to report. We
    locate the peak of the smoothed curve and take the minimum before it, so the
    crossing search only sees the rising phase.
    """
    if len(values) == 0:
        return 0
    smoothed = smooth(values)
    peak_idx = int(np.argmax(smoothed))
    return int(np.argmin(smoothed[: peak_idx + 1]))


def first_crossing(values: np.ndarray, threshold: float, start_idx: int) -> int | None:
    """First index at or after `start_idx` whose value exceeds `threshold`."""
    crossing_indices = np.flatnonzero(values[start_idx:] > threshold)
    if len(crossing_indices) == 0:
        return None
    return start_idx + int(crossing_indices[0])


def load_run_stats(run_dir: Path, end_step: float | None):
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
    env_steps = (start + np.arange(len(sr_values)) * interval) * ta

    if end_step is None:
        valid_indices = np.arange(len(sr_values))
    else:
        valid_indices = np.flatnonzero(env_steps <= end_step)

    if len(valid_indices) == 0:
        raise ValueError(f"No train data points found at or before end step {end_step} in {train_dir}")

    valid_sr = sr_values[valid_indices]
    start_idx = rising_start_index(valid_sr)

    steps_to_threshold = {}
    for threshold in THRESHOLDS:
        crossing_idx = first_crossing(valid_sr, threshold, start_idx)
        steps_to_threshold[threshold] = (
            np.nan if crossing_idx is None else env_steps[valid_indices[crossing_idx]]
        )

    final_sr = float(np.mean(valid_sr[-TAIL_POINTS:]))
    return steps_to_threshold, final_sr


def summarize_algo(task_name: str, algo_dir: Path, end_step: float | None):
    run_dirs = sorted(
        d for d in algo_dir.iterdir() if d.is_dir() and d.name.startswith("run")
    )
    if not run_dirs:
        return None

    steps_by_threshold = {threshold: [] for threshold in THRESHOLDS}
    missing_by_threshold = {threshold: [] for threshold in THRESHOLDS}
    final_sr_values = []
    num_runs = 0

    for run_dir in run_dirs:
        try:
            steps_to_threshold, final_sr = load_run_stats(run_dir, end_step=end_step)
        except FileNotFoundError:
            continue
        num_runs += 1
        for threshold in THRESHOLDS:
            steps = steps_to_threshold[threshold]
            if np.isnan(steps):
                missing_by_threshold[threshold].append(run_dir.name)
            else:
                steps_by_threshold[threshold].append(steps)
        final_sr_values.append(final_sr)

    if not final_sr_values:
        return None

    summary = {
        "task": task_name,
        "algorithm": algo_dir.name,
        "num_runs": num_runs,
    }
    for threshold in THRESHOLDS:
        values = steps_by_threshold[threshold]
        summary[f"steps_to_{threshold:g}"] = np.mean(values) / 1e6 if values else np.nan
        summary[f"n_{threshold:g}"] = len(values)
    summary["final_sr"] = np.mean(final_sr_values)

    return summary, missing_by_threshold


def format_table_rows(rows) -> list[list[str]]:
    """Reported table rows: DSRL, Po-Dec, ZPRL per task, steps and sr with two decimals."""
    table_rows = []
    for task in dict.fromkeys(row["task"] for row in rows):
        task_rows = [row for row in rows if row["task"] == task]
        selected = [
            row for algo in TABLE_ALGORITHMS for row in task_rows if row["algorithm"] == algo
        ]
        if not selected:  # ablation tables use their own algorithm names
            selected = sorted(task_rows, key=lambda row: row["algorithm"])
        for row in selected:
            values = [
                task.replace("metaworld_", ""),
                TABLE_ALGO_NAMES.get(row["algorithm"], row["algorithm"]),
            ]
            for threshold in THRESHOLDS:
                steps = row[f"steps_to_{threshold:g}"]
                values.append("N/A" if pd.isna(steps) else f"{steps:.2f}")
            values.append(f"{row['final_sr']:.2f}")
            table_rows.append(values)
    return table_rows


def write_table(rows, out_file: Path) -> None:
    header = ["task", "algorithm"]
    header += [f"steps(M)_to_sr{threshold:g}" for threshold in THRESHOLDS]
    header += ["Final_sr"]
    table_rows = format_table_rows(rows)

    widths = [
        max(len(header[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(header))
    ]
    lines = [", ".join(
        name.ljust(width) if i < 2 else name.rjust(width)
        for i, (name, width) in enumerate(zip(header, widths))
    )]
    for values in table_rows:
        head = ", ".join(value.ljust(width) for value, width in zip(values[:2], widths[:2]))
        # numbers keep the padding of the existing table: no space after the comma
        tail = "".join("," + value.rjust(width) for value, width in zip(values[2:], widths[2:]))
        lines.append(head + tail)
    out_file.write_text("\n".join(lines) + "\n")


def parse_tasks(tasks) -> list[str]:
    """Accept repeated --task flags and/or comma/space separated task lists."""
    parsed = []
    for entry in tasks:
        for task in entry.replace(",", " ").split():
            if task not in parsed:
                parsed.append(task)
    return parsed


@click.command()
@click.option(
    '--task', 'tasks', required=True, multiple=True,
    help='Task(s) to summarize, e.g. --task can --task square, or --task can,square',
)
@click.option(
    '--out', 'out_path', default=str(TABLE_PATH), show_default=True,
    help='Path of the CSV file to write the reported table to.',
)
def main(tasks, out_path):
    """Summarize train stats from processed plot data for one or more tasks."""
    rows = []
    warnings = []

    for task in parse_tasks(tasks):
        task_dir = resolve_task_dir(task)
        end_step = END_STEP_BY_TASK.get(task)

        algo_dirs = sorted(d for d in task_dir.iterdir() if d.is_dir())
        if not algo_dirs:
            click.echo(f"Warning: no algorithm directories found in {task_dir}")
            continue

        for algo_dir in algo_dirs:
            result = summarize_algo(task, algo_dir, end_step=end_step)
            if result is None:
                click.echo(f"Warning: skipped {algo_dir} because no valid run directories were found")
                continue
            summary, missing_by_threshold = result
            rows.append(summary)
            for threshold, missing in missing_by_threshold.items():
                if missing:
                    warnings.append(
                        f"{task}/{algo_dir.name}: never reached {threshold:g} -> {', '.join(missing)}"
                    )

    if not rows:
        click.echo("No valid statistics found.")
        return

    df = pd.DataFrame(rows)
    for threshold in THRESHOLDS:
        column = f"steps_to_{threshold:g}"
        df[column] = df[column].map(lambda x: "N/A" if pd.isna(x) else f"{x:.3f}")
    df["final_sr"] = df["final_sr"].map(lambda x: f"{x:.3f}")

    click.echo(df.to_string(index=False))

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_table(rows, out_file)
    click.echo(f"\nWrote table to {out_file}")

    if warnings:
        click.echo("")
        for warning in warnings:
            click.echo(f"Warning: {warning}")


if __name__ == '__main__':
    main()
