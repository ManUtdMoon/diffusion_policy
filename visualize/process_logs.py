
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
    'square': { # robomimic-square
        'Ta4': [
            'data/outputs-till-20260912/2026.08.21/13.08.42_train_online_robomimic_workspace_square_image',
            'data/outputs-till-20260912/2026.08.21/13.08.44_train_online_robomimic_workspace_square_image',
            'data/outputs-till-20260912/2026.08.21/13.08.46_train_online_robomimic_workspace_square_image'
        ],
        'Ta4+stage_rew': [
            'data/outputs-till-20260912/2026.08.21/21.15.47_train_online_robomimic_workspace_square_image',
            'data/outputs-till-20260912/2026.08.21/21.15.49_train_online_robomimic_workspace_square_image',
            'data/outputs-till-20260912/2026.08.21/21.15.51_train_online_robomimic_workspace_square_image'
        ],
        'Tr2+stage_rew': [
            'data/outputs/2026.09.14/15.50.40_train_online_robomimic_workspace_square_image',
            'data/outputs/2026.09.14/15.50.42_train_online_robomimic_workspace_square_image',
            'data/outputs/2026.09.14/15.50.44_train_online_robomimic_workspace_square_image'
        ],
        'V2 + 4-step return': [
            'data/outputs/2026.09.22/20.04.44_train_online_reactive_robomimic_workspace_square_image',
            'data/outputs/2026.09.22/20.04.46_train_online_reactive_robomimic_workspace_square_image',
            'data/outputs/2026.09.22/20.04.48_train_online_reactive_robomimic_workspace_square_image'
        ]
    },
}

# =============================================================================
# 2. Metric and Path Configuration
# =============================================================================
METRIC_CONFIG = {
    "train": {"key": "info/recent_done_sr"},
    "eval": {"key": "test/mean_score"}
}
BASE_INPUT_DIR = Path("./")
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
            interval = config["training"].get('training_freq')
            start = config["training"].get('learning_start', 0)
        else:  # mode == 'eval'
            interval = config["training"].get('eval_every')
            start = 0
        Ta = config.get('n_rl_steps', config.get('n_action_steps', 4))
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


class MultiValueOptionCommand(click.Command):
    """Expand grouped --algo/--mode values into repeated Click options."""

    grouped_options = {'--algo', '--mode'}

    def parse_args(self, ctx, args):
        expanded_args = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg not in self.grouped_options:
                expanded_args.append(arg)
                i += 1
                continue

            i += 1
            if i == len(args) or args[i].startswith('--'):
                expanded_args.append(arg)
                continue

            while i < len(args) and not args[i].startswith('--'):
                expanded_args.extend([arg, args[i]])
                i += 1

        return super().parse_args(ctx, expanded_args)


@click.command(cls=MultiValueOptionCommand)
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
    Example: --algo Po-dec ZPRL --mode train eval
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
