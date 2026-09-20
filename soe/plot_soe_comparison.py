from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = {
    'metaworld_box-close': {
        'ZPRL': ROOT / 'data/outputs/zprl_mw_box-close/success_rates.csv',
        'SOE': ROOT / 'data/soe/metaworld_box-close/success_rates.csv',
    },
    'adroit_hammer': {
        'ZPRL': ROOT / 'data/outputs/zprl_adroit_scale_grid_20260920_002901/adroit_hammer/scale_0.4/success_rates.csv',
        'SOE': ROOT / 'data/soe/adroit_hammer/success_rates.csv',
    },
}
OUTPUT_DIR = ROOT / 'data/plot/soe_comparison'
PALETTE = {'ZPRL': '#ff4000', 'SOE': '#4f95cd'}


def main():
    sns.set_theme(style='whitegrid')
    arial_path = ROOT / 'data/Arial.ttf'
    font_manager.fontManager.addfont(str(arial_path))
    mpl.rcParams['font.family'] = font_manager.FontProperties(fname=str(arial_path)).get_name()
    mpl.rcParams['pdf.fonttype'] = 42
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for task, paths in DATA_PATHS.items():
        frames = []
        for algorithm, path in paths.items():
            frame = pd.read_csv(path)
            frame['Episodes'] = frame['round'] * 250
            frame['Algorithm'] = algorithm
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.lineplot(
            data=data, x='Episodes', y='success_rate',
            hue='Algorithm', hue_order=['ZPRL', 'SOE'], palette=PALETTE,
            linewidth=3, errorbar=('ci', 95), seed=0,
            err_kws={'alpha': 0.1, 'linewidth': 0, 'edgecolor': 'none'}, ax=ax,
        )
        ax.set_xlabel('# of Episodes', fontsize=14)
        ax.set_ylabel('Success Rate', fontsize=14)
        ax.set_title(task.capitalize(), fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.yaxis.set_major_formatter('{x:.1f}')
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_xlim(0, 1000)
        ax.set_ylim(0.4, 1.01)
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='lightgrey')
        ax.legend(title=None, fontsize=14, loc='lower right')
        fig.tight_layout()
        for extension in ('pdf', 'png'):
            output = OUTPUT_DIR / f'{task}_soe_comparison.{extension}'
            fig.savefig(output, dpi=300, bbox_inches='tight')
            print(output)
        plt.close(fig)


if __name__ == '__main__':
    main()
