import click
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from pathlib import Path

# =============================================================================
# Plotting Configuration
# =============================================================================
LINEWIDTH = 2
PLOT_CONFIGS = {
    'default': {
        'palette': {
            'ZPRL': '#c66a42',
            'Po-dec': '#696fa2',
            'DSRL': '#e8cd81',
            'ReinFlow': '#89c085',
            'Offline': '#000000',
            'DPPO': '#8aaeb2',
        },
        'algo_name_map': {
            'ZPRL': 'ZPRL',
            'Po-dec': 'Po-dec',
            'DSRL': 'DSRL',
            'ReinFlow': 'ReinFlow',
            'Offline': 'Offline',
            'DPPO': 'DPPO',
        },
    },
    'num_demo': {
        'palette': {
            '25': '#e8cd81',
            '50': '#d59f5e',
            '75': '#c66a42',
            '100': '#984322',
        },
        'algo_name_map': {
            '25': r'$N_\mathrm{demo}=25$',
            '50': r'$N_\mathrm{demo}=50$',
            '75': r'$N_\mathrm{demo}=75$',
            '100': r'$N_\mathrm{demo}=100$',
        },
    },
    'dim': {
        'palette': {
            '8': '#696fa2',
            '16': '#c66a42',
            '32': '#e8cd81',
        },
        'algo_name_map': {
            '8': r'$\mathrm{dim}(z)=8$',
            '16': r'$\mathrm{dim}(z)=16$',
            '32': r'$\mathrm{dim}(z)=32$',
        },
    },
    'scale': {
        'palette': {
            '0.15': '#e8cd81',
            '0.175': '#89c085',
            '0.2': '#c66a42',
            '0.225': '#696fa2',
            '0.25': '#6a4a2b',
        },
        'algo_name_map': {
            '0.15': r'$\lambda=0.15$',
            '0.175': r'$\lambda=0.175$',
            '0.2': r'$\lambda=0.2$',
            '0.225': r'$\lambda=0.225$',
            '0.25': r'$\lambda=0.25$',
        },
    },
}

DS = 20  # Downsample factor
CI_ALPHA = 0.1

LINEWIDTHS = {
    'ZPRL': 3,
    'Po-dec': 1.5,
    'DSRL': 1.5,
    'ReinFlow': 1.5,
    'Offline': 1.5,
}

OFFLINE = {
    'can': 0.8,
    'square': 0.42,
    'transport': 0.62,
}


# Font configuration
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
ARIAL_FONT_PATH = DATA_DIR / 'Arial.ttf'
CAMBRIA_MATH_FONT_PATH = DATA_DIR / 'cambria-math.ttf'
STIX_MATH_FONT_PATH = DATA_DIR / 'STIXTwoMath.otf'

for font_path in [ARIAL_FONT_PATH, CAMBRIA_MATH_FONT_PATH, STIX_MATH_FONT_PATH]:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))

if ARIAL_FONT_PATH.exists():
    mpl.rcParams['font.family'] = font_manager.FontProperties(
        fname=str(ARIAL_FONT_PATH)
    ).get_name()
else:
    mpl.rcParams['font.family'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']

math_font_name = 'Cambria Math'
if CAMBRIA_MATH_FONT_PATH.exists():
    math_font_name = font_manager.FontProperties(
        fname=str(CAMBRIA_MATH_FONT_PATH)
    ).get_name()

mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = math_font_name
mpl.rcParams['mathtext.it'] = math_font_name
mpl.rcParams['mathtext.bf'] = math_font_name
mpl.rcParams['mathtext.cal'] = math_font_name
mpl.rcParams['mathtext.sf'] = math_font_name
mpl.rcParams['mathtext.fallback'] = 'stix'


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
    task_key = task
    plot_config = PLOT_CONFIGS.get(task_key, PLOT_CONFIGS['default'])
    palette = plot_config['palette']
    algo_name_map = plot_config['algo_name_map']
    hue_order = list(algo_name_map.keys())

    base_dir = Path("data/plot") / task_key
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
        algo_key = algo_dir.name
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
            if algo_key == 'ReinFlow':
                smooth_factor = 1
            elif mode == 'train':
                smooth_factor = 3
            else: # mode == 'eval'
                smooth_factor = 1
            y_values = smooth(y_values, sm=smooth_factor)
            
            # Calculate x-axis steps
            num_points = len(y_values)
            x_steps = (start + np.arange(num_points) * interval) * ta_multiplier / 1e6

            ds = DS
            if mode == 'eval' or algo_key == 'ReinFlow':
                ds = 1
            x_steps = x_steps[::ds]
            y_values = y_values[::ds]

            # Create DataFrame for this run
            run_df = pd.DataFrame({
                'Steps': x_steps,
                'Performance': y_values,
                'Algorithm': algo_key,
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
        hue_order=hue_order,
        palette=palette,
        errorbar=('ci', 95),
        err_kws={'alpha': CI_ALPHA, 'linewidth': 0, 'edgecolor': 'none'},
        ax=ax,
        linewidth=LINEWIDTH,
    )

    offline_value = OFFLINE.get(task_key)
    if offline_value is not None:
        ax.axhline(
            y=offline_value,
            color='black',
            linewidth=LINEWIDTHS['Offline'],
            linestyle='--',
        )

    # --- Customize Aesthetics ---
    ax.set_xlabel(r'Env Steps ($\times 10^6$)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    if task_key in ['can', 'square', 'transport']:
        task = 'robomimic_' + task_key
    elif task_key in ['num_demo', 'dim', 'scale']:
        task = 'robomimic_square_' + task_key
    else:
        task = task_key
    ax.set_title(f'{task.capitalize()}', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.yaxis.set_major_formatter('{x:.1f}')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='lightgrey')
    
    # Set reasonable limits, e.g., based on data range
    ax.set_xlim(left=0)
    if 'can' in task:
        ax.set_xlim(right=1)
    elif 'square' in task:
        ax.set_xlim(right=3)
    elif 'transport' in task:
        ax.set_xlim(right=3)
    else:
        raise ValueError(f"Unknown task: {task}")
    ax.set_ylim(bottom=0.0, top=1.01)

    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    legend_labels = [algo_name_map.get(label, label) for label in labels]
    legend = ax.legend(handles=handles, labels=legend_labels, fontsize=10, title='Algorithm')
    legend.set_title('')
    # ax.get_legend().remove()  # remove legend
    plt.tight_layout()

    output_filename = base_dir / f'{task}_{mode}.pdf'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_filename}")


if __name__ == '__main__':
    main()
