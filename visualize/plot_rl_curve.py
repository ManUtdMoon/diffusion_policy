import click
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Plotting Configuration
# =============================================================================
PALETTE = {
    'ZPRL': '#E63983',
    'Po-dec': '#56B4E9',
    'DSRL': '#59D171',
    'ReinFlow': '#F6AA4A',
    'Offline': '#9A4DFF',
    'DPPO': '#A0522D',
}

DS = 10  # Downsample factor

LINEWIDTHS = {
    'ZPRL': 3,
    'Po-dec': 1.5,
    'DSRL': 1.5,
    'ReinFlow': 1.5,
    'DPPO': 1.5,
    'Offline': 1.5,
}

def smooth(data, sm=2):
    """Simple moving average smoothing."""
    if sm > 1:
        z = np.ones_like(data)
        y = np.ones(sm) * 1.0
        d = np.convolve(y, data, "same") / np.convolve(y, z, "same")
    else:
        d = data
    return d

@click.command()
@click.option('--task', required=True, help='Task name to plot (e.g., "square").')
@click.option('--mode', required=True, type=click.Choice(['train', 'eval']), help='Mode to plot: "train" or "eval".')
def main(task, mode):
    """
    Generates and saves a performance plot for a given task and mode.

    This script reads processed data from the 'data/plot/' directory,
    aggregates results across different runs for each algorithm, and
    plots the mean performance with a 95% confidence interval.
    The final plot is saved as '<task>_<mode>.pdf'.
    """
    base_dir = Path("data/plot") / task
    if not base_dir.is_dir():
        print(f"Error: Task directory not found at {base_dir}")
        return

    all_data = []
    algo_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    algo_dirs.sort()
    if not algo_dirs:
        print(f"No algorithm directories found in {base_dir}")
        return

    print(f"Searching for data in: {base_dir}")
    for algo_dir in algo_dirs:
        algo_name = algo_dir.name
        run_dirs = [d for d in algo_dir.iterdir() if d.is_dir() and d.name.startswith('run')]

        for run_dir in run_dirs:
            mode_dir = run_dir / mode
            if not mode_dir.is_dir():
                continue

            # Check for required files
            sr_path = mode_dir / "sr.csv"
            interval_path = mode_dir / "interval.txt"
            start_path = mode_dir / "start.txt"
            ta_path = mode_dir / "Ta.txt"  # New Ta.txt for x-axis scaling

            if not all(p.exists() for p in [sr_path, interval_path]):
                print(f"Warning: Missing data files in {mode_dir}. Skipping.")
                continue

            # Load data
            try:
                y_values = pd.read_csv(sr_path, header=None)[0].values
                interval = int(interval_path.read_text())
                start = int(start_path.read_text()) if start_path.exists() else 0
                # Default Ta to 1.0 if Ta.txt is not present
                ta_multiplier = float(ta_path.read_text()) if ta_path.exists() else 1.0
            except (ValueError, pd.errors.EmptyDataError) as e:
                print(f"Warning: Could not process files in {mode_dir}. Error: {e}. Skipping.")
                continue

            if len(y_values) == 0:
                continue

            # Smooth y-values
            non_smooth_tasks = ['door', 'hammer', 'pen']
            if (
                (algo_name == 'ReinFlow' and task == 'can')
                # (algo_name == 'DPPO' and task in non_smooth_tasks) or
                # (algo_name == 'ReinFlow' and task in non_smooth_tasks) or
                # (algo_name == 'DSRL' and task in non_smooth_tasks)
            ):
                smooth_factor = 1
            elif mode == 'train':
                smooth_factor = 5
            else: # mode == 'eval'
                smooth_factor = 2
            y_values = smooth(y_values, sm=smooth_factor)
            
            # Calculate x-axis steps
            num_points = len(y_values)
            x_steps = (start + np.arange(num_points) * interval) * ta_multiplier / 1e6

            ds = DS
            if mode == 'eval' or (task == 'can' and algo_name == 'ReinFlow'):
                ds = 1
            x_steps = x_steps[::ds]
            y_values = y_values[::ds]

            # Create DataFrame for this run
            run_df = pd.DataFrame({
                'Steps': x_steps,
                'Performance': y_values,
                'Algorithm': algo_name,
                'Run': run_dir.name,
            })
            all_data.append(run_df)

    if not all_data:
        print("Error: No valid data found to plot.")
        return

    # Concatenate all run data into a single DataFrame
    df = pd.concat(all_data, ignore_index=True)
    print("\nSuccessfully loaded data for algorithms:", df['Algorithm'].unique().tolist())

    # =========================================================================
    # Plotting with Seaborn
    # =========================================================================
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.lineplot(
        data=df,
        x='Steps',
        y='Performance',
        hue='Algorithm',
        palette=PALETTE,
        errorbar=('ci', 95),
        ax=ax,
        size='Algorithm',
        sizes=LINEWIDTHS,
    )

    # --- Customize Aesthetics ---
    ax.set_xlabel(r'Env Steps ($\times 10^6$)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title(f'{task.capitalize()}', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.yaxis.set_major_formatter('{x:.2f}')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='lightgrey')
    
    # Set reasonable limits, e.g., based on data range
    ax.set_xlim(left=0)
    if task == 'can':
        ax.set_xlim(right=5)
    elif task == 'square':
        ax.set_xlim(right=8)
    elif task == 'transport':
        ax.set_xlim(right=10)
    elif task == 'door':
        ax.set_xlim(right=2)
    elif task == 'hammer':
        ax.set_xlim(right=1)
    elif task == 'pen':
        ax.set_xlim(right=2)
    elif task.startswith('metaworld'):
        ax.set_xlim(right=1)
    ax.set_ylim(bottom=-0.05, top=1.05)

    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles=handles, labels=labels, fontsize=10, title='Algorithm')
    legend.set_title('')
    ax.get_legend().remove()  # remove legend
    plt.tight_layout()

    output_filename = base_dir / f'{task}_{mode}.pdf'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_filename}")


if __name__ == '__main__':
    main()
