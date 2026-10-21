"""Independent OneLeg steering evaluation; import Isaac Gym before torch."""

import json
from pathlib import Path

import click


@click.command()
@click.option('-c', '--checkpoint', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), required=True)
@click.option('-d', '--device', default='cuda:0', show_default=True)
@click.option('--n-rollouts', type=click.IntRange(min=1), default=100)
@click.option('--n-envs', type=click.IntRange(min=1), default=20)
@click.option('--max-steps', type=click.IntRange(min=1), default=700)
@click.option('--seed', type=int, default=10000, help='First environment seed.')
@click.option('--policy-seed', type=int, default=0, show_default=True, help='FM/noise random generator seed.')
@click.option('--deterministic/--stochastic', default=None,
    help='Default: deterministic residual for ResRL, stochastic noise for DSRL.')
@click.option('--record-video/--no-record-video', default=None)
@click.option('--n-vis', type=click.IntRange(min=0), default=5)
@click.option('--graphics-device-id', type=click.IntRange(min=0), default=None)
def main(checkpoint, output_dir, device, n_rollouts, n_envs, max_steps, seed,
        policy_seed, deterministic, record_video, n_vis, graphics_device_id):
    import isaacgym
    import torch
    from zprl.common.oneleg_online import load_oneleg_online
    from zprl.env_runner.oneleg_online_runner import OneLegOnlineRunner

    device = torch.device(device)
    if device.type != 'cuda':
        raise click.BadParameter('OneLeg simulation requires a CUDA device', param_hint='--device')
    policy, cfg, metadata = load_oneleg_online(checkpoint, device=device)
    evaluation = cfg.online_task.evaluation
    settings = dict(
        n_rollouts=evaluation.n_rollouts if n_rollouts is None else n_rollouts,
        n_envs=evaluation.n_envs if n_envs is None else n_envs,
        max_steps=cfg.online_task.max_steps if max_steps is None else max_steps,
        seed=evaluation.seed if seed is None else seed,
        record_video=evaluation.record_video if record_video is None else record_video,
        n_vis=evaluation.n_vis if n_vis is None else n_vis,
        n_action_steps=cfg.n_rl_steps if metadata['method'] == 'resrl' else cfg.n_action_steps,
        render_size=list(cfg.online_task.render_size), randomness=cfg.online_task.randomness,
        graphics_device_id=graphics_device_id)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runner = OneLegOnlineRunner(output, **settings)
    metrics = runner.run(policy, deterministic=deterministic, policy_seed=policy_seed)
    result = dict(metadata=metadata, metrics=metrics, episodes=runner.last_results,
        checkpoint=str(Path(checkpoint).resolve()), policy_seed=policy_seed,
        deterministic=runner.deterministic, **settings)
    (output / 'eval_log.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    click.echo(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
