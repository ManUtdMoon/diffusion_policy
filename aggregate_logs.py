import json
from pathlib import Path

import pandas as pd
import yaml


logs_list = [
    "data/outputs/2026.06.23/15.01.15_train_online_noise_robomimic_workspace_tool_hang_image_abs",
    "data/outputs/2026.06.23/15.01.19_train_online_noise_robomimic_workspace_tool_hang_image_abs",
    "data/outputs/2026.06.23/15.01.21_train_online_noise_robomimic_workspace_tool_hang_image_abs"
]

STEP_KEY = "info/global_step"
OUTPUT_ROOT = Path("data/outputs")


def read_exp_name(run_dir: Path) -> str:
    config_path = run_dir / ".hydra" / "config.yaml"
    with config_path.open() as f:
        config = yaml.safe_load(f)
    return config["exp_name"]


def read_logs(run_dir: Path) -> pd.DataFrame:
    log_path = run_dir / "logs.json.txt"
    records = []
    with log_path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {log_path}:{line_number}") from e

    frame = pd.DataFrame(records)
    if STEP_KEY not in frame.columns:
        raise KeyError(f"{log_path} does not contain {STEP_KEY!r}")
    if frame[STEP_KEY].duplicated().any():
        raise ValueError(f"{log_path} contains duplicate {STEP_KEY!r}")
    return frame.sort_values(STEP_KEY).reset_index(drop=True)


def main():
    if not logs_list:
        raise ValueError("logs_list is empty")

    run_dirs = [Path(path) for path in logs_list]
    exp_names = [read_exp_name(run_dir) for run_dir in run_dirs]
    if len(set(exp_names)) != 1:
        raise ValueError(f"Runs have different exp_name values: {exp_names}")

    frames = [read_logs(run_dir) for run_dir in run_dirs]
    reference_steps = frames[0][STEP_KEY].tolist()
    reference_columns = set(frames[0].columns)
    for run_dir, frame in zip(run_dirs[1:], frames[1:]):
        if frame[STEP_KEY].tolist() != reference_steps:
            raise ValueError(
                f"{run_dir} has different {STEP_KEY!r} values")
        if set(frame.columns) != reference_columns:
            raise ValueError(f"{run_dir} has different log fields")

    combined = pd.concat(frames, ignore_index=True)
    numeric_columns = combined.select_dtypes(include="number").columns
    non_numeric_columns = reference_columns - set(numeric_columns)
    if non_numeric_columns:
        raise TypeError(
            f"Cannot average non-numeric fields: {sorted(non_numeric_columns)}")
    averaged = (
        combined[numeric_columns]
        .groupby(STEP_KEY, as_index=False, sort=True)
        .mean()
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{exp_names[0]}.csv"
    averaged.to_csv(output_path, index=False)
    print(f"Saved {len(averaged)} steps from {len(run_dirs)} runs to {output_path}")


if __name__ == "__main__":
    main()
