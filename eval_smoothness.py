import hashlib
import json
import math
import os
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Any, Optional

import click
import dill
import gym
import h5py
import hydra
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
import robomimic.utils.file_utils as FileUtils

from zprl.common.pytorch_util import dict_apply
from zprl.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from zprl.env_runner.robomimic_image_runner import create_env
from zprl.gym_util.async_vector_env import AsyncVectorEnv
from zprl.gym_util.multistep_wrapper import MultiStepWrapper
from zprl.model.common.rotation_transformer import RotationTransformer
from zprl.policy.latent_policy import SumPolicy as LatentSumPolicy
from zprl.policy.residue_policy import SumPolicy as ActionSumPolicy


TRACE_ACTION_POS_KEY = 'smoothness/action_pos'
TRACE_STATE_POS_KEY = 'smoothness/state_pos'
TRACE_CHECKSUM_KEY = 'smoothness/initial_state_checksum'
DATASET_ROOT = './data_local/robomimicv030/'
SUCCESS_THRESHOLD = 0.9
DERIVATIVE_NAMES = (
    'action_velocity',
    'action_acceleration',
    'action_jerk',
    'state_velocity',
    'state_acceleration',
    'state_jerk',
)


@dataclass
class EvalPolicyBundle:
    policy: Any
    policy_type: str
    cfg: DictConfig
    env_runner_cfg: DictConfig
    shape_meta: DictConfig
    task_name: str
    n_obs_steps: int
    n_action_steps: int


@dataclass
class TrajectoryRecord:
    seed: int
    success: bool
    kept_chunks: int
    primitive_step_count: int
    initial_state_checksum: Optional[str]
    action_pos: np.ndarray
    state_pos: np.ndarray


def _load_payload(checkpoint: str):
    with open(checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)
    cfg_yaml = OmegaConf.to_yaml(payload['cfg'])
    if 'diffusion_policy' in cfg_yaml:
        payload['cfg'] = OmegaConf.create(
            cfg_yaml.replace('diffusion_policy', 'zprl'))
    return payload


def load_eval_policy(
        policy_type: str,
        checkpoint: str,
        device: str,
        base_ckpt: Optional[str] = None,
        ) -> EvalPolicyBundle:
    if policy_type not in ('base', 'zprl', 'resrl'):
        raise ValueError(f"Unknown policy type: {policy_type}")
    device = torch.device(device)

    def load_base(payload, n_action_steps=None, num_inference_steps=None):
        cfg = payload['cfg']
        if n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
            cfg.policy.n_action_steps = n_action_steps
            cfg.task.env_runner.n_action_steps = n_action_steps
            cfg.task.dataset.pad_after = n_action_steps - 1
        if num_inference_steps is not None:
            cfg.policy.num_inference_steps = num_inference_steps
        state_key = 'ema_model' \
            if n_action_steps is not None or bool(cfg.training.use_ema) else 'model'
        policy = hydra.utils.instantiate(cfg.policy)
        policy.load_state_dict(payload['state_dicts'][state_key])
        policy.to(device)
        policy.eval()
        policy.requires_grad_(False)
        return policy, cfg

    payload = _load_payload(checkpoint)
    cfg = payload['cfg']
    if policy_type == 'base':
        if base_ckpt is not None:
            raise ValueError("--base-ckpt is only valid for zprl or resrl")
        policy, cfg = load_base(payload)
        env_runner_cfg = cfg.task.env_runner
    else:
        target = str(cfg.res_policy._target_)
        expected = 'latent_policy' if policy_type == 'zprl' else 'residue_policy'
        if expected not in target:
            raise ValueError(
                f"--policy-type {policy_type} does not match {target}")

        base_ckpt = base_ckpt or str(cfg.online_task.base_ckpt)
        base_payload = _load_payload(base_ckpt)
        if str(base_payload['cfg'].task_name) != str(cfg.task_name):
            raise ValueError(
                f"Base task {base_payload['cfg'].task_name} does not match "
                f"{cfg.task_name}")

        Ta = int(cfg.n_action_steps)
        To = int(cfg.n_obs_steps)
        base_policy, _ = load_base(
            base_payload,
            n_action_steps=Ta,
            num_inference_steps=int(getattr(cfg, 'num_inference_steps', 2)),
        )
        do = int(base_policy.obs_feature_dim)
        da = int(cfg.shape_meta.action.shape[0])

        if policy_type == 'zprl':
            res_policy = hydra.utils.instantiate(
                cfg.res_policy,
                obs_dim=To * do,
                z_dim=int(base_policy.vib_latent_dim),
                action_dim=Ta * da,
            )
            policy = LatentSumPolicy(
                res_scale=cfg.training.res_scale,
                base_policy=base_policy,
                res_policy=res_policy,
            )
        else:
            res_policy = hydra.utils.instantiate(
                cfg.res_policy, obs_dim=To * do, action_dim=Ta * da)
            policy = ActionSumPolicy(
                res_scale=cfg.training.res_scale,
                obs_emb_dim=do,
                action_dim=da,
                n_action_steps=Ta,
                base_policy=base_policy,
                res_policy=res_policy,
            )

        res_policy.load_state_dict(payload['res_policy'])
        res_policy.to(device)
        res_policy.eval()
        res_policy.requires_grad_(False)
        policy.eval()
        env_runner_cfg = cfg.online_task.env_runner

    if not bool(env_runner_cfg.abs_action):
        raise ValueError("Only absolute-action tasks are supported")
    if tuple(cfg.shape_meta.action.shape) != (10,):
        raise ValueError("Only single-arm action shape (10,) is supported")
    if tuple(cfg.shape_meta.obs.robot0_eef_pos.shape) != (3,):
        raise ValueError("robot0_eef_pos must have shape (3,)")

    return EvalPolicyBundle(
        policy=policy,
        policy_type=policy_type,
        cfg=cfg,
        env_runner_cfg=env_runner_cfg,
        shape_meta=cfg.shape_meta,
        task_name=str(cfg.task_name),
        n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),
    )


def _checksum(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


class PrimitiveTraceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_state_pos = None
        self.initial_state_checksum = None
        self.chunk_action_pos = None
        self.chunk_state_pos = None

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.current_state_pos = np.asarray(
            obs['robot0_eef_pos']).copy()
        state_owner = getattr(self.env, 'env', self.env)
        get_state = getattr(state_owner, 'get_state', None)
        initial_state = get_state()['states'] \
            if callable(get_state) else self.current_state_pos
        self.initial_state_checksum = _checksum(initial_state)
        return obs

    def start_chunk(self):
        self.chunk_action_pos = []
        self.chunk_state_pos = []

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.current_state_pos = np.asarray(
            obs['robot0_eef_pos']).copy()
        self.chunk_action_pos.append(np.asarray(action[:3]).copy())
        self.chunk_state_pos.append(self.current_state_pos)
        return obs, reward, done, info

    def finish_chunk(self):
        trace = {
            TRACE_ACTION_POS_KEY: np.asarray(self.chunk_action_pos),
            TRACE_STATE_POS_KEY: np.asarray(self.chunk_state_pos),
            TRACE_CHECKSUM_KEY: self.initial_state_checksum,
        }
        self.chunk_action_pos = None
        self.chunk_state_pos = None
        return trace


class TraceMultiStepWrapper(MultiStepWrapper):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        current = self.env
        while not isinstance(current, PrimitiveTraceWrapper):
            current = current.env
        self.trace_wrapper = current

    def step(self, action):
        self.trace_wrapper.start_chunk()
        obs, reward, done, info = super().step(action)
        info = dict(info)
        info.update(self.trace_wrapper.finish_chunk())
        return obs, reward, done, info


def _make_env_pool(bundle: EvalPolicyBundle, n_envs: int):
    configured_path = pathlib.Path(
        os.path.expanduser(str(bundle.env_runner_cfg.dataset_path)))
    task_cfg = bundle.cfg.task \
        if bundle.policy_type == 'base' else bundle.cfg.online_task
    local_path = pathlib.Path(DATASET_ROOT).joinpath(
        str(task_cfg.task_name),
        str(task_cfg.dataset_type),
        configured_path.name,
    )
    dataset_path = local_path if local_path.is_file() else configured_path
    if not dataset_path.is_file():
        raise ValueError(f"Dataset not found: {dataset_path}")

    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))
    env_meta['env_kwargs']['use_object_obs'] = False
    env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    control_frequency = float(env_meta['env_kwargs']['control_freq'])

    shape_meta = bundle.shape_meta
    render_obs_key = str(bundle.env_runner_cfg.render_obs_key)
    max_steps = int(bundle.env_runner_cfg.max_steps)
    n_obs_steps = bundle.n_obs_steps
    n_action_steps = bundle.n_action_steps

    def make_env(enable_render):
        robomimic_env = create_env(
            env_meta=env_meta,
            shape_meta=shape_meta,
            enable_render=enable_render,
        )
        robomimic_env.env.hard_reset = False
        image_env = RobomimicImageWrapper(
            env=robomimic_env,
            shape_meta=shape_meta,
            init_state=None,
            render_obs_key=render_obs_key,
        )
        return TraceMultiStepWrapper(
            PrimitiveTraceWrapper(image_env),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
            reward_agg_method='max',
        )

    def env_fn():
        return make_env(True)

    def dummy_env_fn():
        return make_env(False)

    return (
        AsyncVectorEnv([env_fn] * n_envs, dummy_env_fn=dummy_env_fn),
        dataset_path,
        control_frequency,
        max_steps,
    )


def _append_chunk_traces(buffers, rewards, dones, infos):
    for buffer, reward, done, info in zip(buffers, rewards, dones, infos):
        if not buffer['recording']:
            continue
        action_pos = np.asarray(info[TRACE_ACTION_POS_KEY])
        state_pos = np.asarray(info[TRACE_STATE_POS_KEY])
        assert action_pos.shape == state_pos.shape
        buffer['action_pos'].append(action_pos)
        buffer['state_pos'].append(state_pos)
        buffer['kept_chunks'] += 1
        buffer['initial_state_checksum'] = info[TRACE_CHECKSUM_KEY]
        if float(reward) > SUCCESS_THRESHOLD:
            buffer['success'] = True
            buffer['recording'] = False
        elif done:
            buffer['recording'] = False


def _rollout_batch(
        env,
        policy,
        seeds,
        n_envs,
        max_steps,
        n_action_steps,
        policy_seed,
        rotation_transformer,
        ):
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    torch.cuda.manual_seed_all(policy_seed)
    padded_seeds = list(seeds) + [seeds[0]] * (n_envs - len(seeds))
    env.seed(padded_seeds)
    obs = env.reset()
    policy.reset()

    initial_state_pos = obs['robot0_eef_pos'][:len(seeds), -1]
    buffers = [{
        'seed': int(seed),
        'recording': True,
        'success': False,
        'kept_chunks': 0,
        'action_pos': [],
        'state_pos': [initial_state_pos[i:i + 1].copy()],
        'initial_state_checksum': None,
    } for i, seed in enumerate(seeds)]
    episode_done = np.zeros(n_envs, dtype=bool)

    for _ in tqdm.tqdm(
            range(math.ceil(max_steps / n_action_steps) + 1),
            desc=f"Rollout seeds {seeds[0]}..{seeds[-1]}",
            mininterval=1.0,
            ):
        obs_dict = dict_apply(
            dict(obs),
            lambda x: torch.from_numpy(x).to(device=policy.device),
        )
        with torch.no_grad():
            action = policy.predict_action(obs_dict)['action']
        action = action.detach().cpu().numpy()
        if not np.all(np.isfinite(action)):
            raise ValueError("Policy produced NaN or Inf action")

        pos = action[..., :3]
        rot = rotation_transformer.inverse(action[..., 3:9])
        env_action = np.concatenate([pos, rot, action[..., [-1]]], axis=-1)
        obs, rewards, dones, infos = env.step(env_action)

        count = len(seeds)
        _append_chunk_traces(
            buffers,
            rewards[:count],
            dones[:count],
            infos[:count],
        )
        episode_done[:count] |= dones[:count]
        if np.all(episode_done[:count]):
            break
    else:
        raise RuntimeError("Episodes did not finish within max_steps")

    records = []
    for buffer in buffers:
        step_count = sum(len(x) for x in buffer['action_pos'])
        if buffer['success']:
            action_pos = np.concatenate(buffer['action_pos'])
            state_pos = np.concatenate(buffer['state_pos'])
            assert len(state_pos) == len(action_pos) + 1
        else:
            action_pos = np.empty((0, 3), dtype=np.float32)
            state_pos = np.empty((0, 3), dtype=np.float32)
        records.append(TrajectoryRecord(
            seed=buffer['seed'],
            success=buffer['success'],
            kept_chunks=buffer['kept_chunks'],
            primitive_step_count=step_count,
            initial_state_checksum=buffer['initial_state_checksum'],
            action_pos=action_pos,
            state_pos=state_pos,
        ))
    return records


def rollout_policy(bundle, seed_start, n_seeds, n_envs, policy_seed):
    env, dataset_path, control_frequency, max_steps = _make_env_pool(
        bundle, n_envs)
    seeds = list(range(seed_start, seed_start + n_seeds))
    rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
    records = []
    try:
        n_batches = math.ceil(n_seeds / n_envs)
        for batch_idx in range(n_batches):
            start = batch_idx * n_envs
            batch_seeds = seeds[start:start + n_envs]
            click.echo(
                f"Rollout batch {batch_idx + 1}/{n_batches}: "
                f"seeds {batch_seeds[0]}..{batch_seeds[-1]}")
            records.extend(_rollout_batch(
                env,
                bundle.policy,
                batch_seeds,
                n_envs,
                max_steps,
                bundle.n_action_steps,
                policy_seed + batch_idx,
                rotation_transformer,
            ))
    finally:
        env.close()
    return {
        'records': records,
        'seeds': seeds,
        'dataset_path': str(dataset_path),
        'control_frequency': control_frequency,
        'dt': 1.0 / control_frequency,
        'max_steps': max_steps,
    }


def compute_trajectory_derivatives(record, dt):
    empty = np.empty((0, 3), dtype=np.float64)
    if not record.success:
        return {name: empty.copy() for name in DERIVATIVE_NAMES}
    action_velocity = np.diff(record.action_pos, axis=0) / dt
    action_acceleration = np.diff(action_velocity, axis=0) / dt
    state_velocity = np.diff(record.state_pos, axis=0) / dt
    state_acceleration = np.diff(state_velocity, axis=0) / dt
    return {
        'action_velocity': action_velocity,
        'action_acceleration': action_acceleration,
        'action_jerk': np.diff(action_acceleration, axis=0) / dt,
        'state_velocity': state_velocity,
        'state_acceleration': state_acceleration,
        'state_jerk': np.diff(state_acceleration, axis=0) / dt,
    }


def _sample_stats(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)

    def scalar(values):
        if len(values) == 0:
            return {'mean': None, 'std': None}
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }

    return {
        'sample_count': len(vectors),
        'norm': scalar(np.linalg.norm(vectors, axis=-1)),
        'x': scalar(vectors[:, 0]),
        'y': scalar(vectors[:, 1]),
        'z': scalar(vectors[:, 2]),
    }


def aggregate_statistics(records, dt):
    successful = [record for record in records if record.success]
    derivatives = [
        compute_trajectory_derivatives(record, dt)
        for record in successful
    ]
    metrics = {}
    for name in DERIVATIVE_NAMES:
        samples = [item[name] for item in derivatives if len(item[name])]
        pooled = np.concatenate(samples) \
            if samples else np.empty((0, 3), dtype=np.float64)
        trajectory_means = np.asarray([
            np.linalg.norm(item[name], axis=-1).mean()
            for item in derivatives
            if len(item[name])
        ])
        trajectory_mean = float(np.mean(trajectory_means)) \
            if len(trajectory_means) else None
        trajectory_std = float(np.std(trajectory_means, ddof=1)) \
            if len(trajectory_means) > 1 else None
        metrics[name] = {
            'pooled_sample': _sample_stats(pooled),
            'trajectory_level': {
                'trajectory_count': len(trajectory_means),
                'mean_of_trajectory_means': trajectory_mean,
                'std_of_trajectory_means': trajectory_std,
            },
        }
    return {
        'n_requested': len(records),
        'n_success': len(successful),
        'success_rate': len(successful) / len(records),
        'successful_seeds': [record.seed for record in successful],
        'metrics': metrics,
    }


def write_outputs(output_dir, metadata, summary, records, dt):
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_dir.joinpath('metadata.json').open('w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    with output_dir.joinpath('summary.json').open('w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with output_dir.joinpath('per_seed.jsonl').open('w') as f:
        for record in records:
            metrics = None
            if record.success:
                metrics = {
                    name: _sample_stats(values)
                    for name, values in
                    compute_trajectory_derivatives(record, dt).items()
                }
            json.dump({
                'seed': record.seed,
                'success': record.success,
                'kept_chunks': record.kept_chunks,
                'primitive_step_count': record.primitive_step_count,
                'initial_state_checksum': record.initial_state_checksum,
                'metrics': metrics,
            }, f, sort_keys=True)
            f.write('\n')

    with h5py.File(output_dir.joinpath('trajectories.hdf5'), 'w') as f:
        for record in records:
            if not record.success:
                continue
            group = f.create_group(f'seed_{record.seed}')
            group.create_dataset(
                'action_pos', data=record.action_pos, compression='gzip')
            group.create_dataset(
                'state_pos', data=record.state_pos, compression='gzip')
            group.attrs['kept_chunks'] = record.kept_chunks
            group.attrs['primitive_step_count'] = record.primitive_step_count
            group.attrs[
                'initial_state_checksum'] = record.initial_state_checksum


@click.command()
@click.option(
    '--policy-type',
    required=True,
    type=click.Choice(['base', 'zprl', 'resrl'], case_sensitive=True),
)
@click.option('-c', '--checkpoint', required=True, type=click.Path(dir_okay=False))
@click.option('-o', '--output-dir', required=True, type=click.Path(file_okay=False))
@click.option('-d', '--device', default='cuda:0', show_default=True)
@click.option('-b', '--base-ckpt', default=None, type=click.Path(dir_okay=False))
@click.option('--seed-start', default=100000, show_default=True, type=int)
@click.option('--n-seeds', default=150, show_default=True, type=click.IntRange(min=1))
@click.option('--n-envs', default=50, show_default=True, type=click.IntRange(min=1))
@click.option('--policy-seed', default=0, show_default=True, type=int)
def main(
        policy_type,
        checkpoint,
        output_dir,
        device,
        base_ckpt,
        seed_start,
        n_seeds,
        n_envs,
        policy_seed,
        ):
    output_dir = pathlib.Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        click.confirm(
            f"Output path {output_dir} is not empty. Overwrite result files?",
            abort=True)

    bundle = load_eval_policy(
        policy_type, checkpoint, device, base_ckpt)
    click.echo(
        f"Loaded {policy_type} checkpoint for task {bundle.task_name}: "
        f"To={bundle.n_obs_steps}, Ta={bundle.n_action_steps}")
    result = rollout_policy(
        bundle, seed_start, n_seeds, n_envs, policy_seed)
    summary = aggregate_statistics(result['records'], result['dt'])

    digest = hashlib.sha256()
    with open(checkpoint, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    configured_base = OmegaConf.select(
        bundle.cfg, 'online_task.base_ckpt', default=None)
    used_base = base_ckpt or configured_base
    metadata = {
        'policy_type': policy_type,
        'checkpoint': str(pathlib.Path(checkpoint).resolve()),
        'checkpoint_sha256': digest.hexdigest(),
        'base_checkpoint': (
            str(pathlib.Path(str(used_base)).resolve())
            if used_base is not None else None),
        'task_name': bundle.task_name,
        'dataset_path': str(pathlib.Path(result['dataset_path']).resolve()),
        'seeds': result['seeds'],
        'seed_start': seed_start,
        'n_seeds': n_seeds,
        'n_envs': n_envs,
        'policy_seed': policy_seed,
        'n_obs_steps': bundle.n_obs_steps,
        'n_action_steps': bundle.n_action_steps,
        'max_steps': result['max_steps'],
        'control_frequency_hz': result['control_frequency'],
        'dt_seconds': result['dt'],
        'action_representation': 'absolute',
        'action_position_slice': [0, 3],
        'state_position_key': 'robot0_eef_pos',
        'success_threshold': SUCCESS_THRESHOLD,
        'success_chunk_policy': 'keep_full_chunk',
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'device': device,
        'command': sys.argv,
    }
    write_outputs(
        output_dir, metadata, summary, result['records'], result['dt'])
    click.echo(
        f"Saved smoothness evaluation to {output_dir}: "
        f"{summary['n_success']}/{summary['n_requested']} successful.")


if __name__ == '__main__':
    main()
