import json
import pathlib
import random
import sys
from collections import deque
from functools import partial

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import click
import numpy as np
import torch

from soe.soe_util import SOE_TASKS, SoeExplorationPolicy, load_soe_policy
from zprl.gym_util.async_vector_env import AsyncVectorEnv
from zprl.gym_util.multistep_wrapper import repeated_space, stack_last_n_obs


def make_env(task, cfg, render_device_id):
    if task == 'adroit_hammer':
        from zprl.env.adroit.adroit import AdroitEnv
        env = AdroitEnv(env_name=cfg.task.env_runner.task_name,
            render_device_id=render_device_id)
        max_steps = cfg.task.env_runner.max_steps // 2
    else:
        from zprl.env.metaworld.metaworld_image_wrapper import MetaWorldEnv
        env = MetaWorldEnv(task_name=cfg.task.env_runner.task_name,
            device_id=render_device_id)
        max_steps = cfg.task.env_runner.max_steps
    return env, max_steps


class CollectionEnv:
    """Record and save each episode inside its environment worker."""
    def __init__(self, env, task, max_steps, n_obs_steps, n_action_steps):
        self.env = env
        self.task = task
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.metadata = env.metadata
        self.observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.action_space = repeated_space(env.action_space, n_action_steps)

    def start_episode(self, reset_seed, output_path):
        self.env.seed(reset_seed)
        self.output_path = output_path

    def reset(self):
        self.obs = self.env.reset()
        self.history = deque([self.obs], maxlen=self.n_obs_steps)
        self.data = {key: [] for key in (
            'img', 'state', 'action', 'reward', 'success_signal', 'terminated', 'truncated')}
        if self.task == 'adroit_hammer':
            self.data['n_goal_achieved'] = []
            self.data['accumulated_goal_achieved'] = []
        return self._get_obs()

    def _get_obs(self):
        return {key: stack_last_n_obs(
            [item[key] for item in self.history], self.n_obs_steps) for key in self.obs}

    def step(self, actions):
        data = self.data
        reward_sum = 0.
        for action in actions:
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high).astype(np.float32)
            data['img'].append(np.rint(np.moveaxis(self.obs['image'], 0, -1) * 255).clip(0, 255).astype(np.uint8))
            data['state'].append(self.obs['agent_pos'].astype(np.float32, copy=True))
            data['action'].append(action.copy())
            self.obs, reward, env_done, info = self.env.step(action)
            self.history.append(self.obs)
            reward_sum += reward
            truncated = bool(info.get('TimeLimit.truncated', False))
            terminated = bool(env_done) and not truncated
            if not terminated and len(data['action']) >= self.max_steps:
                truncated = True
            done = terminated or truncated
            data['reward'].append(reward)
            data['terminated'].append(terminated)
            data['truncated'].append(truncated)
            if self.task == 'adroit_hammer':
                data['n_goal_achieved'].append(info['n_goal_achieved'])
                data['accumulated_goal_achieved'].append(info['accumulated_goal_achieved'])
                data['success_signal'].append(info['n_goal_achieved'] > 0)
            else:
                data['success_signal'].append(bool(info['success']))
            if done:
                break
        result = {}
        if done:
            result = {
                'success': bool(np.any(data['success_signal'])),
                'length': len(data['action']),
                'termination_reason': 'timeout' if truncated else 'terminated',
            }
            if self.task == 'adroit_hammer':
                count = int(data['accumulated_goal_achieved'][-1])
                result['accumulated_goal_achieved'] = count
                result['success'] = count >= 50
            # AsyncVectorEnv resets on done, so save before returning to its worker.
            np.savez_compressed(self.output_path, **data)
        return self._get_obs(), reward_sum, done, result

    def close(self):
        self.env.close()


def make_collection_env(task, cfg, render_device_id, n_obs_steps, n_action_steps):
    env, max_steps = make_env(task, cfg, render_device_id)
    return CollectionEnv(env, task, max_steps, n_obs_steps, n_action_steps)


def collect_rollouts(checkpoint, task, seed, round, num_episodes=250,
        exploration_alpha=2.0, device='cuda:0', render_device_id=None, output=None, n_envs=25):
    assert num_episodes % n_envs == 0, 'num_episodes must be divisible by n_envs'
    output = pathlib.Path(output) if output is not None else (
        pathlib.Path('data/soe') / task / f'seed_{seed}' / f'round_{round}' / 'rollouts')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'episodes').mkdir()
    device = torch.device(device)
    if render_device_id is None:
        render_device_id = device.index if device.index is not None else 0
    policy, cfg, weights_key = load_soe_policy(checkpoint, task, device)
    config = {
        'task': task, 'seed': seed, 'round': round,
        'checkpoint': str(pathlib.Path(checkpoint).resolve()),
        'weights_key': weights_key, 'num_episodes': num_episodes, 'n_envs': n_envs,
        'exploration_alpha': exploration_alpha,
        'vib_encoder_alpha': policy.vib_encoder.alpha,
        'device': str(device), 'render_device_id': render_device_id,
        'n_obs_steps': policy.n_obs_steps, 'n_action_steps': policy.n_action_steps,
        'num_inference_steps': policy.num_inference_steps,
        'max_steps': int(cfg.task.env_runner.max_steps) // (2 if task == 'adroit_hammer' else 1),
        'action_repeat': 2 if task == 'adroit_hammer' else 1,
    }
    with (output / 'collection_config.json').open('w') as f:
        json.dump(config, f, indent=2, sort_keys=True)
    summary = dict(num_episodes=0, success_episodes=0, failed_episodes=0,
        timeout_episodes=0, env_steps=0)
    env_fn = partial(make_collection_env, task, cfg, render_device_id,
        policy.n_obs_steps, policy.n_action_steps)
    exploration_policy = SoeExplorationPolicy(policy, exploration_alpha)
    envs = AsyncVectorEnv([env_fn] * n_envs, dummy_env_fn=env_fn)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with (output / 'manifest.jsonl').open('w') as manifest:
            for start in range(0, num_episodes, n_envs):
                indices = list(range(start, start + n_envs))
                seeds = [np.random.SeedSequence([seed, round, i]).generate_state(2) for i in indices]
                reset_seeds = [int(s[0]) | (1 << 31) for s in seeds]
                paths = [pathlib.Path('episodes') / f'ep_{i:06d}.npz' for i in indices]
                envs.call_each('start_episode', args_list=[
                    (reset_seed, str((output / path).resolve()))
                    for reset_seed, path in zip(reset_seeds, paths)])
                obs = envs.reset()
                exploration_policy.reset()
                policy_seed = int(seeds[0][1])
                torch.manual_seed(policy_seed)
                np.random.seed(policy_seed)
                random.seed(policy_seed)
                while True:
                    obs_dict = {key: torch.from_numpy(value).to(
                        device=policy.device, dtype=policy.dtype) for key, value in obs.items()}
                    actions = exploration_policy.predict_action(obs_dict)['action'].cpu().numpy()
                    obs, _, dones, results = envs.step(actions)
                    if np.any(dones):
                        assert np.all(dones), 'Parallel SOE collection expects equal episode lengths'
                        break
                for batch_index, (i, reset_seed, path, result) in enumerate(
                        zip(indices, reset_seeds, paths, results)):
                    result.update({
                        'episode_uid': f'{task}/{seed}/{round}/{i}',
                        'task': task, 'seed': seed, 'round': round, 'episode_index': i,
                        'reset_seed': reset_seed, 'policy_seed': policy_seed,
                        'batch_start': start, 'batch_index': batch_index,
                        'checkpoint': config['checkpoint'],
                        'exploration_alpha': exploration_alpha,
                        'trajectory_path': str(path),
                    })
                    manifest.write(json.dumps(result, sort_keys=True) + '\n')
                    summary['num_episodes'] += 1
                    summary['success_episodes'] += int(result['success'])
                    summary['failed_episodes'] += int(not result['success'])
                    summary['timeout_episodes'] += int(result['termination_reason'] == 'timeout')
                    summary['env_steps'] += result['length']
                manifest.flush()
                click.echo(f"Collected {start + n_envs}/{num_episodes} episodes, "
                    f"{summary['success_episodes']} successful.")
    finally:
        envs.close()
    with (output / 'collection_summary.json').open('w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


@click.command()
@click.option('-c', '--checkpoint', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--task', required=True, type=click.Choice(SOE_TASKS))
@click.option('--seed', required=True, type=click.IntRange(0, 2**32 - 1))
@click.option('--round', required=True, type=click.IntRange(1, 4))
@click.option('--num-episodes', default=250, type=click.IntRange(1), show_default=True)
@click.option('--n-envs', default=25, type=click.IntRange(1), show_default=True)
@click.option('--exploration-alpha', default=2.0, type=click.FloatRange(min=0), show_default=True)
@click.option('-d', '--device', default='cuda:0', show_default=True)
@click.option('--render-device-id', default=None, type=click.IntRange(0))
@click.option('-o', '--output', default=None, type=click.Path(file_okay=False),
    help='Defaults to data/soe/TASK/seed_SEED/round_ROUND/rollouts/.')
def main(**kwargs):
    """Collect complete SOE exploration episodes, including failures."""
    summary = collect_rollouts(**kwargs)
    click.echo(f"Collected {summary['num_episodes']} episodes, "
        f"{summary['success_episodes']} successful.")


if __name__ == '__main__':
    main()
