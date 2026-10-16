"""Standalone OneLeg evaluation; Isaac Gym is imported before torch at startup."""

import json
import math
import pathlib

import click


@click.command()
@click.option('-c', '--checkpoint', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), required=True)
@click.option('-d', '--device', default='cuda:0', show_default=True)
@click.option('--n-rollouts', default=100, type=click.IntRange(min=1), show_default=True)
@click.option('--n-envs', default=20, type=click.IntRange(min=1), show_default=True)
@click.option('--max-steps', default=700, type=click.IntRange(min=1), show_default=True)
@click.option('-t', '--n-action-steps', default=None, type=click.IntRange(1, 16))
@click.option('-n', '--num-inference-steps', default=None, type=click.IntRange(min=1))
@click.option('--seed', default=10000, type=int, show_default=True,
    help='First environment seed; rollout i resets with seed + i.')
@click.option('--record-video/--no-record-video', default=False, show_default=True)
@click.option('--n-vis', default=5, type=click.IntRange(min=0), show_default=True,
    help='Maximum number of rollout videos across the entire evaluation.')
@click.option('--render-size', type=(click.IntRange(min=1), click.IntRange(min=1)),
    default=(640, 360), show_default=True, help='Native camera width and height.')
@click.option('--graphics-device-id', default=None, type=click.IntRange(min=0))
def main(checkpoint, output_dir, device, n_rollouts, n_envs, max_steps,
        n_action_steps, num_inference_steps, seed, record_video, n_vis, graphics_device_id, render_size):
    if n_rollouts % n_envs:
        raise click.BadParameter('must be divisible by --n-envs', param_hint='--n-rollouts')
    import isaacgym
    import dill
    import hydra
    from omegaconf import OmegaConf
    import torch
    from zprl.env_runner.oneleg_image_runner import OneLegImageRunner

    OmegaConf.register_new_resolver('eval', eval, replace=True)
    device = torch.device(device)
    if device.type != 'cuda':
        raise click.BadParameter('OneLeg simulation requires a CUDA device', param_hint='--device')
    with open(checkpoint, 'rb') as file:
        payload = torch.load(file, map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    if cfg.task_name != 'oneleg_image':
        raise click.ClickException(f'Expected oneleg_image checkpoint, got {cfg.task_name}')
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = 'ema_model' if cfg.training.use_ema else 'model'
    policy.load_state_dict(payload['state_dicts'][state_key])
    policy = policy.to(device).requires_grad_(False).eval()
    del payload
    if num_inference_steps is not None:
        policy.num_inference_steps = num_inference_steps
    if n_action_steps is None:
        n_action_steps = policy.n_action_steps
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runner = OneLegImageRunner(output_dir=output, n_rollouts=n_rollouts,
        n_envs=n_envs, max_steps=max_steps, n_action_steps=n_action_steps,
        seed=seed, record_video=record_video, n_vis=n_vis, graphics_device_id=graphics_device_id,
        render_size=render_size)
    metrics = runner.run(policy)
    metrics = {key: None if isinstance(value, float) and math.isnan(value) else value
               for key, value in metrics.items()}
    result = dict(metrics=metrics, episodes=runner.last_results,
        checkpoint=str(pathlib.Path(checkpoint).resolve()), n_action_steps=n_action_steps,
        num_inference_steps=policy.num_inference_steps, seed=seed,
        n_envs=n_envs, max_steps=max_steps, record_video=record_video, n_vis=n_vis,
        render_size=list(render_size))
    with (output / 'eval_log.json').open('w') as file:
        json.dump(result, file, indent=2, allow_nan=False)
    click.echo(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
