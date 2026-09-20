import json
import pathlib
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import click
import numpy as np
import zarr

from soe.soe_util import SOE_TASKS
from zprl.common.replay_buffer import ReplayBuffer


def build_success_dataset(task, seed, through_round, rollout_root=None, output=None):
    rollout_root = pathlib.Path(rollout_root) if rollout_root is not None else (
        pathlib.Path('data/soe') / task / f'seed_{seed}')
    output = pathlib.Path(output) if output is not None else (
        rollout_root / f'round_{through_round}' / 'success_dataset' / 'dataset.zarr')
    selected = []
    rounds = []
    for round in range(1, through_round + 1):
        manifest_path = rollout_root / f'round_{round}' / 'rollouts' / 'manifest.jsonl'
        with manifest_path.open() as f:
            episodes = sorted((json.loads(line) for line in f),
                key=lambda episode: episode['episode_index'])
        stats = {'round': round, 'attempted_episodes': 0, 'success_episodes': 0}
        for episode in episodes:
            if (episode['task'], episode['seed'], episode['round']) != (task, seed, round):
                raise ValueError(f"Episode does not belong to this task/seed/round: {episode['episode_uid']}")
            stats['attempted_episodes'] += 1
            if episode['success']:
                stats['success_episodes'] += 1
                selected.append({
                    **episode,
                    'manifest_path': str(manifest_path.resolve()),
                    'trajectory_path': str((manifest_path.parent / episode['trajectory_path']).resolve()),
                    'dataset_episode_index': len(selected),
                })
        rounds.append(stats)
    if not selected:
        raise ValueError(f"No successful episodes for {task}, seed {seed}, through round {through_round}")

    output.mkdir(parents=True, exist_ok=False)
    # Append directly to disk instead of holding all rounds' images in RAM.
    replay = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(output)))
    for episode in selected:
        with np.load(episode['trajectory_path']) as data:
            replay.add_episode({key: data[key] for key in ('img', 'state', 'action')})
    summary = {
        'task': task, 'seed': seed, 'through_round': through_round,
        'rounds': rounds,
        'cumulative_attempted_episodes': sum(r['attempted_episodes'] for r in rounds),
        'cumulative_success_episodes': replay.n_episodes,
        'cumulative_transitions': int(replay.n_steps),
        'dataset_path': str(output.resolve()),
    }
    with (output.parent / 'lineage.json').open('w') as f:
        json.dump(selected, f, indent=2, sort_keys=True)
    with (output.parent / 'dataset_summary.json').open('w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


@click.command()
@click.option('--task', required=True, type=click.Choice(SOE_TASKS))
@click.option('--seed', required=True, type=click.IntRange(0, 2**32 - 1))
@click.option('--through-round', required=True, type=click.IntRange(1, 4))
@click.option('--rollout-root', default=None, type=click.Path(file_okay=False),
    help='Defaults to data/soe/TASK/seed_SEED/.')
@click.option('-o', '--output', default=None, type=click.Path(file_okay=False),
    help='Defaults to ROLLOUT_ROOT/round_ROUND/success_dataset/dataset.zarr.')
def main(**kwargs):
    """Build cumulative SOE imitation data from successful collected episodes."""
    summary = build_success_dataset(**kwargs)
    click.echo(f"Saved {summary['cumulative_success_episodes']} successful episodes "
        f"({summary['cumulative_transitions']} transitions) to {summary['dataset_path']}")


if __name__ == '__main__':
    main()
