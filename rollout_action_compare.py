import json
import math
import pathlib
import random

import click
import numpy as np
import torch
import tqdm

from eval_smoothness import (
    SUCCESS_THRESHOLD,
    _make_env_pool,
    load_eval_policy,
)
from zprl.common.pytorch_util import dict_apply
from zprl.model.common.rotation_transformer import RotationTransformer


SEED_AXES = {
    100016: 'x',
    100088: 'y',
    100081: 'z',
}
AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


def rollout(method, checkpoint, output_dir, device, base_ckpt, policy_seed):
    policy_type = 'resrl' if method == 'podec' else 'zprl'
    bundle = load_eval_policy(
        policy_type, checkpoint, device, base_ckpt)
    seeds = list(SEED_AXES)
    env, dataset_path, control_frequency, max_steps = _make_env_pool(
        bundle, len(seeds), ema_weight=0.0)

    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    torch.cuda.manual_seed_all(policy_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env.seed(seeds)
    obs = env.reset()
    bundle.policy.reset()
    rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
    records = {
        seed: {
            'base_action': [],
            'sum_action': [],
            'success_chunk': None,
        }
        for seed in seeds
    }

    try:
        for chunk_idx in tqdm.tqdm(
                range(math.ceil(max_steps / bundle.n_action_steps) + 1),
                desc=f'{method} rollout',
                mininterval=1.0,
                ):
            obs_dict = dict_apply(
                dict(obs),
                lambda x: torch.from_numpy(x).to(bundle.policy.device),
            )
            with torch.no_grad():
                result = bundle.policy.predict_action(
                    obs_dict, return_base_action=True)
            sum_action = result['action'].detach().cpu().numpy()
            base_action = result['base_action'].detach().cpu().numpy()

            pos = sum_action[..., :3]
            rot = rotation_transformer.inverse(sum_action[..., 3:9])
            gripper = sum_action[..., [-1]]
            env_action = np.concatenate([pos, rot, gripper], axis=-1)
            obs, rewards, dones, _ = env.step(env_action)

            for i, seed in enumerate(seeds):
                record = records[seed]
                if record['success_chunk'] is not None:
                    continue
                axis_idx = AXIS_INDEX[SEED_AXES[seed]]
                record['base_action'].append(
                    base_action[i, :, axis_idx].copy())
                record['sum_action'].append(
                    sum_action[i, :, axis_idx].copy())
                if float(rewards[i]) > SUCCESS_THRESHOLD:
                    record['success_chunk'] = chunk_idx
                elif dones[i]:
                    raise RuntimeError(
                        f'{method} seed {seed} ended before success')

            if all(
                    record['success_chunk'] is not None
                    for record in records.values()):
                break
    finally:
        env.close()

    failed = [
        seed for seed, record in records.items()
        if record['success_chunk'] is None
    ]
    if failed:
        raise RuntimeError(f'{method} failed for seeds: {failed}')

    output_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / control_frequency
    for seed, record in records.items():
        axis = SEED_AXES[seed]
        base_action = np.concatenate(record['base_action'])
        sum_action = np.concatenate(record['sum_action'])
        np.savez_compressed(
            output_dir / f'{method}_seed={seed}_{axis}.npz',
            seed=seed,
            axis=axis,
            dt_seconds=dt,
            time_seconds=np.arange(len(sum_action)) * dt,
            base_action=base_action,
            sum_action=sum_action,
            success_chunk=record['success_chunk'],
        )

    metadata = {
        'method': method,
        'checkpoint': str(pathlib.Path(checkpoint).resolve()),
        'base_checkpoint': (
            str(pathlib.Path(base_ckpt).resolve())
            if base_ckpt is not None else None),
        'dataset_path': str(pathlib.Path(dataset_path).resolve()),
        'seed_axes': SEED_AXES,
        'policy_seed': policy_seed,
        'n_envs': len(seeds),
        'n_action_steps': bundle.n_action_steps,
        'control_frequency_hz': control_frequency,
        'recording': 'base_and_sum_action_through_success_chunk',
    }
    with output_dir.joinpath(f'{method}_metadata.json').open('w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


@click.command()
@click.option(
    '--method', required=True,
    type=click.Choice(['podec', 'zprl'], case_sensitive=True))
@click.option('-c', '--checkpoint', required=True, type=click.Path(dir_okay=False))
@click.option(
    '-o', '--output-dir',
    default='data/plot/action_compare_rollouts',
    show_default=True,
    type=click.Path(file_okay=False),
)
@click.option('-d', '--device', default='cuda:0', show_default=True)
@click.option('-b', '--base-ckpt', default=None, type=click.Path(dir_okay=False))
@click.option('--policy-seed', default=0, show_default=True, type=int)
def main(method, checkpoint, output_dir, device, base_ckpt, policy_seed):
    rollout(
        method=method,
        checkpoint=checkpoint,
        output_dir=pathlib.Path(output_dir),
        device=device,
        base_ckpt=base_ckpt,
        policy_seed=policy_seed,
    )
    click.echo(f'Saved {method} action rollouts to {output_dir}')


if __name__ == '__main__':
    main()
