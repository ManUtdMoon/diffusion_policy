
import click
import yaml
import pandas as pd
from pathlib import Path

# =============================================================================
# 1. Hardcoded dictionary for experiment mapping
# As requested, the dictionary of algorithm names to experiment folders is here.
# You can modify this dictionary to match your experiments.
# Paths are relative to 'data/outputs/'.
# =============================================================================
TASK_TO_ALGO_EXP_MAP = {
    'can': { # robomimic-can
        'Po-dec': [
            '2025.12.05/09.36.02_train_online_robomimic_workspace_can_image',
            '2025.12.05/12.33.40_train_online_robomimic_workspace_can_image',
            '2025.12.05/15.25.39_train_online_robomimic_workspace_can_image',
        ],
        'ZPRL': [
            '2025.12.05/11.13.30_train_online_vib_robomimic_workspace_can_image',
            '2025.12.05/15.00.13_train_online_vib_robomimic_workspace_can_image',
            '2025.12.05/17.53.38_train_online_vib_robomimic_workspace_can_image',
        ],
        'DSRL': [
            '2025.12.05/22.47.33_train_online_noise_robomimic_workspace_can_image',
            '2025.12.06/04.36.49_train_online_noise_robomimic_workspace_can_image',
            '2025.12.06/01.42.29_train_online_noise_robomimic_workspace_can_image',
        ],
    },
    'square': { # robomimic-square
        'Po-dec': [
            '2025.12.05/17.52.08_train_online_robomimic_workspace_square_image',
            '2025.12.05/22.21.16_train_online_robomimic_workspace_square_image',
            '2025.12.06/01.18.44_train_online_robomimic_workspace_square_image',
        ],
        'ZPRL': [
            '2025.12.06/11.37.04_train_online_vib_robomimic_workspace_square_image',
            '2025.12.06/14.40.48_train_online_vib_robomimic_workspace_square_image',
            '2025.12.06/17.39.21_train_online_vib_robomimic_workspace_square_image',
        ],
        'DSRL': [
            '2025.12.06/11.39.00_train_online_noise_robomimic_workspace_square_image',
            '2025.12.06/14.41.13_train_online_noise_robomimic_workspace_square_image',
            '2025.12.06/17.42.44_train_online_noise_robomimic_workspace_square_image',
        ],
    },
    'transport': { # robomimic-transport
        'Po-dec': [
            '2025.12.09/12.56.40_train_online_robomimic_workspace_transport_image',
            '2025.12.09/18.47.08_train_online_robomimic_workspace_transport_image',
            '2025.12.10/00.16.36_train_online_robomimic_workspace_transport_image',
        ],
        'ZPRL': [
            '2025.12.09/16.08.09_train_online_vib_robomimic_workspace_transport_image',
            '2025.12.09/10.05.15_train_online_vib_robomimic_workspace_transport_image',
            '2025.12.10/17.09.40_train_online_vib_robomimic_workspace_transport_image',
        ],
        'DSRL': [
            '2025.12.10/09.50.51_train_online_noise_robomimic_workspace_transport_image',
            '2025.12.10/09.52.47_train_online_noise_robomimic_workspace_transport_image',
            '2025.12.10/15.20.59_train_online_noise_robomimic_workspace_transport_image',
        ],
    },
    'door': { # adroit_door
        'Po-dec': [
            '2025.12.15/15.59.05_train_online_adroit_workspace_adroit_door',
            '2025.12.16/09.52.06_train_online_adroit_workspace_adroit_door',
            '2025.12.16/10.59.43_train_online_adroit_workspace_adroit_door',
        ],
        'ZPRL': [
            '2025.12.15/20.06.12_train_online_vib_adroit_workspace_adroit_door',
            '2025.12.16/09.53.32_train_online_vib_adroit_workspace_adroit_door',
            '2025.12.16/10.58.17_train_online_vib_adroit_workspace_adroit_door',
        ],
        'DSRL': [
            '2026.03.16/19.26.52_train_online_noise_workspace_adroit_door',
            '2026.03.16/19.05.46_train_online_noise_workspace_adroit_door',
            '2026.03.16/17.09.11_train_online_noise_workspace_adroit_door',
        ]
    },
    'hammer' : { # adroit_hammer
        'Po-dec': [
            '2025.12.16/11.28.21_train_online_adroit_workspace_adroit_hammer',
            '2025.12.16/12.25.32_train_online_adroit_workspace_adroit_hammer',
            '2025.12.16/13.01.21_train_online_adroit_workspace_adroit_hammer',
        ],
        'ZPRL': [
            '2025.12.15/20.57.19_train_online_vib_adroit_workspace_adroit_hammer',
            '2025.12.16/14.57.18_train_online_vib_adroit_workspace_adroit_hammer',
            '2025.12.16/16.00.19_train_online_vib_adroit_workspace_adroit_hammer',
        ]
    },
    'pen' : { # adroit_pen
        'Po-dec': [
            '2025.12.16/23.16.10_train_online_adroit_workspace_adroit_pen',
            '2025.12.17/00.17.15_train_online_adroit_workspace_adroit_pen',
            '2025.12.17/01.23.29_train_online_adroit_workspace_adroit_pen',
        ],
        'ZPRL': [
            '2025.12.16/21.35.44_train_online_vib_adroit_workspace_adroit_pen',
            '2025.12.17/01.12.35_train_online_vib_adroit_workspace_adroit_pen',
            '2025.12.17/02.16.25_train_online_vib_adroit_workspace_adroit_pen',
        ],
        'DSRL': [
            '2026.03.16/16.59.18_train_online_noise_workspace_adroit_pen',
            '2026.03.16/18.23.27_train_online_noise_workspace_adroit_pen',
            '2026.03.16/19.25.31_train_online_noise_workspace_adroit_pen',
        ]
    },
    'metaworld_push-wall': {
        'Po-dec': [
            '2025.12.18/18.52.53_train_online_workspace_push-wall',
            '2025.12.18/19.57.15_train_online_workspace_push-wall',
            '2025.12.18/21.01.37_train_online_workspace_push-wall',
        ],
        'ZPRL': [
            '2025.12.19/11.16.42_train_online_vib_workspace_push-wall',
            '2025.12.19/14.19.56_train_online_vib_workspace_push-wall',
            '2025.12.19/16.23.35_train_online_vib_workspace_push-wall',
        ],
    },
    'metaworld_box-close': {
        'Po-dec': [
            '2025.12.18/17.14.18_train_online_workspace_box-close',
            '2025.12.18/22.09.03_train_online_workspace_box-close',
            '2025.12.18/23.14.37_train_online_workspace_box-close',
        ],
        'ZPRL': [
            '2025.12.19/11.20.08_train_online_vib_workspace_box-close',
            '2025.12.19/22.10.47_train_online_vib_workspace_box-close',
            '2025.12.19/23.15.12_train_online_vib_workspace_box-close',
        ],
    },
    'metaworld_': {
    },
}

# =============================================================================
# 2. Metric and Path Configuration
# =============================================================================
METRIC_CONFIG = {
    "train": {"key": "info/recent_done_sr"},
    "eval": {"key": "test/mean_score"}
}
BASE_INPUT_DIR = Path("data/outputs")
BASE_OUTPUT_DIR = Path("data/plot")


def load_metrics(log_file: Path, metric_key: str):
    """
    Load metrics from a logs.json.txt file.

    Args:
        log_file: Path to the logs.json.txt file.
        metric_key: The specific metric to extract from the logs.

    Returns:
        A list of floating-point metric values.
    """
    if not log_file.is_file():
        print(f"Warning: Log file not found: {log_file}")
        return []

    # Each line in the file is a separate JSON object
    df = pd.read_json(log_file, lines=True)

    if metric_key not in df.columns:
        print(f"Warning: Metric key '{metric_key}' not found in {log_file}")
        return []

    metrics = df[metric_key].dropna().tolist()
    return metrics


def process_algo_mode(task, mode, algo, algo_exp_map):
    """
    Processes experiment logs for a single algorithm and mode.
    """
    if algo not in algo_exp_map:
        raise click.BadParameter(f"Algorithm '{algo}' not found for task '{task}'. Available algos: {list(algo_exp_map.keys())}")

    exp_dirs = algo_exp_map[algo]
    metric_key = METRIC_CONFIG[mode]['key']

    print(f"Processing task: '{task}', algorithm: '{algo}', mode: '{mode}'...")

    first_interval = None
    first_start = None
    first_Ta = None

    for i, run_dir_str in enumerate(exp_dirs):
        run_dir = BASE_INPUT_DIR / run_dir_str
        print(f"  - Processing run {i}: {run_dir_str}")

        # 1. Load Hydra config and determine interval/start
        config_path = run_dir / ".hydra" / "config.yaml"
        if not config_path.is_file():
            print(f"    Warning: config.yaml not found in {run_dir / '.hydra'}")
            continue

        with config_path.open('r') as f:
            config = yaml.safe_load(f)

        if mode == 'train':
            interval = config["training"].get('log_every')
            start = config["training"].get('learning_start', 0)
        else:  # mode == 'eval'
            interval = config["training"].get('eval_every')
            start = 0
        Ta = config.get('n_action_steps', 4)
        if task in ['door', 'hammer', 'pen']:
            Ta *= 2  # action repeat

        if interval is None:
            print(f"    Warning: Could not determine interval for mode '{mode}' in {config_path}")
            continue

        # 2. Perform consistency check for interval and start
        if i == 0:
            first_interval = interval
            first_start = start
            first_Ta = Ta
        else:
            if interval != first_interval:
                raise ValueError(
                    f"Inconsistent interval for {algo}/{mode}. "
                    f"Run 0 has interval {first_interval}, but run {i} has {interval}."
                )
            if start != first_start:
                raise ValueError(
                    f"Inconsistent start for {algo}/{mode}. "
                    f"Run 0 has start {first_start}, but run {i} has {start}."
                )
            if Ta != first_Ta:
                raise ValueError(
                    f"Inconsistent Ta for {algo}/{mode}. "
                    f"Run 0 has Ta {first_Ta}, but run {i} has {Ta}."
                )

        # 3. Load metrics from log file
        log_file = run_dir / "logs.json.txt"
        metrics = load_metrics(log_file, metric_key)

        if not metrics:
            print(f"    Warning: No metrics found for run {i}. Skipping file write.")
            continue

        # 4. Create output directory and write files
        output_dir = BASE_OUTPUT_DIR / task / algo / f"run{i}" / mode
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write sr.csv
        pd.Series(metrics).to_csv(output_dir / "sr.csv", index=False, header=False)

        # Write interval.txt
        (output_dir / "interval.txt").write_text(str(interval))

        # Write start.txt
        (output_dir / "start.txt").write_text(str(start))

        # Write Ta.txt
        (output_dir / "Ta.txt").write_text(str(Ta))

        print(f"    Successfully wrote data to {output_dir}")

    print(f"\nProcessing for {algo} {mode} complete.")


@click.command()
@click.option('--task', required=True, help='Name of the task (e.g., can, square).')
@click.option('--mode', required=True, type=click.Choice(['train', 'eval']), help='Mode to process: "train" or "eval".', multiple=True)
@click.option('--algo', required=True, help='Algorithm name to process from the experiment map for the given task.', multiple=True)
def main(task, mode, algo):
    """
    Processes experiment logs to generate standardized metric CSVs and parameter text files.

    This script reads experiment data from 'data/outputs', extracts relevant metrics
    based on the specified mode (train/eval), and saves them in a structured format
    under 'data/plot/<task_name>/<algo_name>/'. It also performs a consistency
    check to ensure all runs for a given algorithm and mode share the same
    interval and start step.

    This script supports processing multiple algorithms and modes in a single run.
    """
    if task not in TASK_TO_ALGO_EXP_MAP:
        raise click.BadParameter(f"Task '{task}' not found. Available tasks: {list(TASK_TO_ALGO_EXP_MAP.keys())}")

    algo_exp_map = TASK_TO_ALGO_EXP_MAP[task]

    for algo_name in algo:
        for mode_name in mode:
            try:
                process_algo_mode(task, mode_name, algo_name, algo_exp_map)
            except Exception as e:
                print(f"Error processing {algo_name}/{mode_name} for task {task}: {e}")

    print("\nBatch processing complete.")


if __name__ == '__main__':
    main()
