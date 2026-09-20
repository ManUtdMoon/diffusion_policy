import json
import pathlib
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import click
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from soe.soe_util import SOE_TASKS, SoeBasePolicy, load_soe_policy


def episode_results(task, runner_log, eval_seeds, success_threshold=None):
    episodes = []
    for i, seed in enumerate(eval_seeds):
        episode = {'episode_id': i, 'reset_seed': seed}
        if task == 'adroit_hammer':
            count = int(runner_log[f'test/n_goal_{seed}'])
            episode['accumulated_goal_achieved'] = count
            episode['success'] = count >= success_threshold
        else:
            episode['success'] = bool(runner_log[f'test/reward_{i}'])
        episodes.append(episode)
    return episodes


def evaluate(checkpoint, task, seed, round, num_episodes=100,
        eval_start_seed=10000, policy_seed=None, n_envs=50, device='cuda:0',
        render_device_id=None, output=None, n_action_steps=None,
        num_inference_steps=None):
    if num_episodes < 1 or n_envs < 1 or num_episodes % n_envs:
        raise ValueError('num_episodes must be positive and divisible by n_envs')
    if not 0 <= round <= 4:
        raise ValueError('round must be between 0 and 4')
    if not 0 <= eval_start_seed <= 2**32 - num_episodes:
        raise ValueError('Evaluation seeds must fit in the uint32 range')
    output = pathlib.Path(output) if output is not None else (
        pathlib.Path('data/soe') / task / f'seed_{seed}' / f'round_{round}' / 'eval')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory is not empty: {output}")
    device = torch.device(device)
    if render_device_id is None:
        render_device_id = device.index if device.index is not None else 0
    if policy_seed is None:
        policy_seed = seed
    policy, cfg, weights_key = load_soe_policy(
        checkpoint, task, device, n_action_steps, num_inference_steps)

    runner_cfg = cfg.task.env_runner
    runner_cfg.eval_episodes = num_episodes
    runner_cfg.test_start_seed = eval_start_seed
    runner_cfg.n_envs = n_envs
    runner_cfg.n_epi_vis = 0
    runner_cfg.n_obs_steps = policy.n_obs_steps
    runner_cfg.n_action_steps = policy.n_action_steps
    runner_cfg.render_device_id = render_device_id
    eval_seeds = list(range(eval_start_seed, eval_start_seed + num_episodes))
    metadata = {
        'task': task,
        'seed': seed,
        'round': round,
        'checkpoint': str(pathlib.Path(checkpoint).resolve()),
        'checkpoint_training_seed': int(cfg.training.seed),
        'weights_key': weights_key,
        'policy_path': 'base',
        'vib_enabled': False,
        'policy_seed': policy_seed,
        'eval_seeds': eval_seeds,
        'device': str(device),
        'n_obs_steps': policy.n_obs_steps,
        'n_action_steps': policy.n_action_steps,
        'num_inference_steps': policy.num_inference_steps,
        'runner': OmegaConf.to_container(runner_cfg, resolve=True),
        'success_rule': ('accumulated_goal_achieved >= 50'
            if task == 'adroit_hammer' else 'any episode success'),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'eval_config.json').open('w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    runner = hydra.utils.instantiate(runner_cfg, output_dir=str(output))
    try:
        # Seed after model/environment construction so their initialization does
        # not consume the policy's evaluation RNG stream.
        torch.manual_seed(policy_seed)
        np.random.seed(policy_seed)
        random.seed(policy_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        runner_log = runner.run(SoeBasePolicy(policy))
        episodes = episode_results(
            task, runner_log, eval_seeds,
            success_threshold=getattr(runner, 'success_threshold', None))
    finally:
        runner.close()

    successes = sum(episode['success'] for episode in episodes)
    summary = {
        key: metadata[key] for key in (
            'task', 'seed', 'round', 'checkpoint', 'weights_key',
            'policy_path', 'vib_enabled', 'policy_seed')
    }
    summary.update({
        'eval_episodes': len(episodes),
        'eval_successes': successes,
        'eval_success_rate': successes / len(episodes),
    })
    with (output / 'eval_episodes.jsonl').open('w') as f:
        for episode in episodes:
            f.write(json.dumps(episode, sort_keys=True) + '\n')
    with (output / 'eval_summary.json').open('w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


@click.command()
@click.option('-c', '--checkpoint', required=True,
    type=click.Path(exists=True, dir_okay=False))
@click.option('--task', required=True, type=click.Choice(SOE_TASKS))
@click.option('--seed', required=True, type=click.IntRange(0, 2**32 - 1),
    help='Experiment seed used in the output directory.')
@click.option('--round', required=True, type=click.IntRange(0, 4),
    help='Policy round; use 0 for the initial checkpoint.')
@click.option('--num-episodes', default=100, type=click.IntRange(1), show_default=True)
@click.option('--eval-start-seed', default=10000, type=click.IntRange(0), show_default=True)
@click.option('--policy-seed', default=None, type=click.IntRange(0, 2**32 - 1),
    help='Policy RNG seed; defaults to --seed.')
@click.option('--n-envs', default=25, type=click.IntRange(1), show_default=True)
@click.option('-d', '--device', default='cuda:0', show_default=True)
@click.option('--render-device-id', default=None, type=click.IntRange(0),
    help='Runner render device; defaults to the policy device index.')
@click.option('-o', '--output', default=None, type=click.Path(file_okay=False),
    help='Defaults to data/soe/TASK/seed_SEED/round_ROUND/eval/.')
@click.option('--n-action-steps', default=None, type=click.IntRange(1),
    help='Override action chunk length; otherwise inherit the checkpoint.')
@click.option('--num-inference-steps', default=None, type=click.IntRange(1),
    help='Override inference steps; otherwise inherit the checkpoint.')
def main(**kwargs):
    """Evaluate an SOE offline checkpoint through its base-only path."""
    try:
        summary = evaluate(**kwargs)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Success rate: {summary['eval_successes']}/"
        f"{summary['eval_episodes']} = {summary['eval_success_rate']:.4f}")


if __name__ == '__main__':
    main()
