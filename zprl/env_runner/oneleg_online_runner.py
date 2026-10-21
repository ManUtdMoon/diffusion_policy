"""Evaluate complete fixed-batch sessions using the online environment contract."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from zprl.env.furniturebench.oneleg_online_env import OneLegOnlineEnv
from zprl.env_runner.base_image_runner import BaseImageRunner


class OneLegOnlineRunner(BaseImageRunner):
    def __init__(self, output_dir, n_rollouts=100, n_envs=20, max_steps=700,
            n_action_steps=8, seed=10000, record_video=False, n_vis=5,
            render_size=(640, 360), randomness='low', graphics_device_id=None):
        super().__init__(output_dir)
        self.n_rollouts = n_rollouts
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.n_action_steps = n_action_steps
        self.seed = seed
        self.record_video = record_video
        self.n_vis = n_vis
        self.render_size = render_size
        self.randomness = randomness
        self.graphics_device_id = graphics_device_id
        self.last_results = []

    def _videos(self, stack, offset, count):
        if not self.record_video:
            return {}
        import imageio.v2 as imageio

        directory = Path(self.output_dir) / 'media'
        directory.mkdir(parents=True, exist_ok=True)
        writers = {}
        for row in range(min(count, max(0, self.n_vis - offset))):
            writer = imageio.get_writer(str(directory / f'rollout_{offset + row:04d}.mp4'),
                                       fps=20 / self.n_action_steps)
            stack.callback(writer.close)
            writers[row] = writer
        return writers

    @staticmethod
    def _write_frames(writers, obs, active):
        for row, writer in writers.items():
            if active[row]:
                frame = torch.cat([obs[key][row, 0] for key in ('color_image1', 'color_image2')], dim=-1)
                frame = frame.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
                writer.append_data(frame)

    @torch.no_grad()
    def run(self, policy, deterministic=None, policy_seed=0):
        is_resrl = hasattr(policy, 'res_policy')
        steering = policy.res_policy if is_resrl else policy.noise_policy
        if deterministic is None:
            deterministic = is_resrl
        self.deterministic = deterministic
        previous_training = steering.training
        policy.eval()
        torch.manual_seed(policy_seed)
        device = next(steering.parameters()).device
        env = None
        solid_steps = idle_steps = 0
        results = []
        try:
            for offset in range(0, self.n_rollouts, self.n_envs):
                count = min(self.n_envs, self.n_rollouts - offset)
                error = None
                try:
                    if env is None:
                        env = OneLegOnlineEnv(n_action_steps=self.n_action_steps,
                            max_steps=self.max_steps, mode='eval', num_envs=self.n_envs,
                            compute_device_id=device.index or 0,
                            graphics_device_id=(device.index or 0) if self.graphics_device_id is None else self.graphics_device_id,
                            render_size=tuple(self.render_size), randomness=self.randomness)
                    obs = env.reset(seed=self.seed + offset)
                except (RuntimeError, FloatingPointError) as exception:
                    error = str(exception)
                    batch_results = [dict(env_index=row, seed=self.seed + offset + row,
                        success=False, termination_reason='failure', episode_steps=0,
                        episode_return=0.) for row in range(self.n_envs)]
                else:
                    policy.reset(obs)
                    with ExitStack() as stack:
                        writers = self._videos(stack, offset, count)
                        self._write_frames(writers, obs, env.active)
                        while not env.batch_done:
                            action = policy.predict_action(obs, deterministic=deterministic)['action']
                            active = env.active.clone()
                            try:
                                obs, rewards, dones, infos = env.step(action)
                            except (RuntimeError, FloatingPointError):
                                if env.batch_error is None:
                                    raise
                                error = env.batch_error
                                break
                            policy.on_env_step(obs, dones, infos)
                            self._write_frames(writers, obs, active)
                    batch_results = env.results
                for row, result in enumerate(batch_results[:count]):
                    result = dict(result, rollout_id=offset + row)
                    if error and result['termination_reason'] == 'failure':
                        result['error'] = error
                    results.append(result)
                if error and env is not None:
                    solid_steps += env.solid_steps
                    idle_steps += env.idle_steps
                    env.close()
                    env = None
        finally:
            if env is not None:
                solid_steps += env.solid_steps
                idle_steps += env.idle_steps
                env.close()
            steering.train(previous_training)
            policy.base_policy.eval()
            self.last_results = results
        successes = [result for result in results if result['success']]
        return dict(n_rollouts=len(results), success_count=len(successes),
            success_rate=len(successes) / len(results),
            timeout_count=sum(result['termination_reason'] == 'timeout' for result in results),
            failure_count=sum(result['termination_reason'] == 'failure' for result in results),
            mean_success_episode_steps=float(np.mean([result['episode_steps'] for result in successes])) if successes else None,
            solid_steps=solid_steps, idle_steps=idle_steps,
            idle_ratio=idle_steps / max(solid_steps + idle_steps, 1))
