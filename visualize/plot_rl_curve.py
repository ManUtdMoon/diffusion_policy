# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path

TASK = "square"  # robomimic: can, square, transport
MODE = "eval" # "train" or "eval"
MODE_PLOT_PARAMS = {
    "train": {
        "key": "info/recent_done_sr",
        "interval": 1000,
        "start": 1000,
        "downsample": 10,
    },
    "eval": {
        "key": "test/mean_score",
        "interval": 50000,
        "start": 0,
        "downsample": 1,
    }
}

# =============================================================================
# 1. Helper function to retrieve eval data
# This part is the same as before.
# =============================================================================
def load_data(run_dir, sm=3):
    """
    Load eval data from run_dir/logs.json.txt

    return:
        np.ndarray: Array of all scores extracted from the log file. (no downsampling)
    """
    log_file = Path(run_dir) / 'logs.json.txt'
    if not log_file.is_file():
        print(f"Log file not found: {log_file}")
        return np.array([])

    scores = []
    with log_file.open('r') as f:
        for line in f:
            try:
                log_data = json.loads(line)
                key = MODE_PLOT_PARAMS[MODE]['key']
                if key in log_data:
                    scores.append(log_data[key])
            except json.JSONDecodeError:
                # Ignore malformed lines
                continue

    return smooth(np.array(scores), sm=sm)


def smooth(data, sm=2):
    '''
    Borrow from
    https://blog.csdn.net/qq_43280087/article/details/119894398
    '''
    if sm > 1:
        z = np.ones_like(data)
        y = np.ones(sm)*1.0
        d = np.convolve(y, data, "same")/np.convolve(y, z, "same")
    else:
        d = data
    return d


# =============================================================================
# 2. Data Preparation using Pandas
# =============================================================================

# Data sources dictionary remains the same
data_sources = {
    'ResRL': [
        '2025.11.25/20.05.52_train_online_robomimic_workspace_square_image',
    ],
    'ZPRL': [
        '2025.11.25/20.03.35_train_online_vib_robomimic_workspace_square_image',
    ],
    'DSRL': [
        '2025.12.01/13.57.44_train_online_noise_robomimic_workspace_square_image',
    ],
}

# --- Create a "long-form" DataFrame from all experimental data ---
## assume all runs shares the same train_step & eval_every, now 1M, 50K
INTERVAL = MODE_PLOT_PARAMS[MODE]['interval']
START = MODE_PLOT_PARAMS[MODE]['start']
DS = MODE_PLOT_PARAMS[MODE]['downsample']
TOTAL_STEPS = 1_000_000
x_steps = np.arange(START, TOTAL_STEPS + 1, INTERVAL)
x_steps = x_steps[::DS] / float(TOTAL_STEPS)  # in M

all_data = []
for i, (name, dirs) in enumerate(data_sources.items()):
    for j, run_dir in enumerate(dirs):
        run_dir = 'data/outputs/' + run_dir
        smooth_factor = 2 if MODE == 'eval' else 9
        y_values = load_data(run_dir, sm=smooth_factor)[::DS]

        assert len(x_steps) == len(y_values), \
            f"Length mismatch: x_steps({len(x_steps)}) vs y_values({len(y_values)}) in {run_dir}"

        # Create a temporary DataFrame for this single run
        run_df = pd.DataFrame({
            'Steps': x_steps,
            'Performance': y_values,
            'Algorithm': name,
            'Run': j
        })
        all_data.append(run_df)


# Concatenate all run data into a single DataFrame
df = pd.concat(all_data, ignore_index=True)

# print(df.head()) # Uncomment to inspect the DataFrame

# =============================================================================
# 3. Plotting with Seaborn and Matplotlib
# =============================================================================

# --- Plotting Configuration ---
# Use the same color and marker schemes
palette = {
    'ZPRL': '#E63983',
    'ResRL': '#56B4E9',
    'DSRL': '#59D171',
}
markers = {
    'random_env': 'o',
    'demo_env': 'v',
    'pi-dec': 's',
    'can_blue_triangle': '^',
    'can_blue_diamond': 'D',
    # backup: x, +, X, *
}

# --- Create Plot ---
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(4.5, 3.5))

# --- Use seaborn.lineplot ---
# This single function handles grouping, aggregation (mean), and error bands (std dev).
sns.lineplot(
    data=df,
    x='Steps',
    y='Performance',
    hue='Algorithm',      # Color lines by algorithm
    style='Algorithm',    # Change markers by algorithm
    # markers=markers,      # Dictionary mapping algorithms to markers
    palette=palette,      # Dictionary mapping algorithms to colors
    dashes=False,
    errorbar=('ci', 95),        # Show standard deviation as the error band
    ax=ax,
    linewidth=1.5,
)

# --- Customize Aesthetics (same as before) ---
# ax.set_title('can', fontsize=14)
ax.set_xlabel('Steps (x4M)', fontsize=12)
ax.set_ylabel('Success rate', fontsize=12)
ax.set_xlim(left=-0.05, right=1.05)
ax.set_ylim(bottom=-0.05, top=1.05)
ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
ax.set_title(f'{TASK}', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.yaxis.set_major_formatter('{x:.2f}')
ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='lightgrey')

# Customize the legend generated by Seaborn
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, fontsize=10)

plt.tight_layout()
fig.savefig(f'{TASK}_{MODE}.pdf', dpi=300, bbox_inches='tight')