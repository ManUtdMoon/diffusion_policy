import logging
import pathlib
from collections import deque
from contextlib import ExitStack
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from zprl.env.furniturebench.oneleg_env import OneLegEnv
from zprl.env_runner.base_image_runner import BaseImageRunner


logger = logging.getLogger(__name__)


class OneLegImageRunner(BaseImageRunner):
    def __init__(self, output_dir, n_rollouts=100, n_envs=20, max_steps=700,
            n_action_steps=8, seed=10000, record_video=False, fps=20,
            graphics_device_id=None, n_vis=5, render_size=(640, 360)):
        super().__init__(output_dir)
        if min(n_rollouts, n_envs, max_steps, fps) < 1 or not 1 <= n_action_steps <= 16:
            raise ValueError('Invalid OneLeg rollout budget or action prefix length')
        if n_rollouts % n_envs:
            raise ValueError('n_rollouts must be divisible by n_envs')
        if n_vis < 0:
            raise ValueError('n_vis must be nonnegative')
        self.n_rollouts = n_rollouts
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.n_action_steps = n_action_steps
        self.seed = seed
        self.record_video = record_video
        self.n_vis = n_vis
        self.fps = fps
        self.graphics_device_id = graphics_device_id
        self.render_size = tuple(render_size)
        self.last_results = []

    def _open_videos(self, offset, count, video_dir):
        import imageio.v2 as imageio

        writers, paths = [], []
        try:
            for i in range(count):
                path = video_dir / f'rollout_{offset + i:04d}.mp4'
                writer = imageio.get_writer(str(path), format='FFMPEG',
                    fps=self.fps, codec='libx264', pixelformat='yuv420p')
                writers.append(writer)
                paths.append(str(path))
        except Exception:
            for writer in writers:
                writer.close()
            raise
        return writers, paths

    @staticmethod
    def _write_frames(writers, obs, active):
        for i, writer in enumerate(writers):
            if active[i]:
                frame = torch.cat([obs['color_image1'][i], obs['color_image2'][i]], dim=2)
                frame = frame.permute(1, 2, 0).mul(255).round().clamp(0, 255).byte().cpu().numpy()
                writer.append_data(np.ascontiguousarray(frame))

    @staticmethod
    def _close_env(env):
        if env is not None:
            try:
                env.close()
            except RuntimeError:
                logger.exception('Failed to close OneLeg simulator')

    def _run_batch(self, env, policy, obs, records, video_dir):
        policy.reset()
        actions = deque()
        active = torch.ones(self.n_envs, dtype=torch.bool, device=policy.device)
        n_videos = min(self.n_envs, max(0, self.n_vis - records[0]['rollout'])) \
            if self.record_video else 0
        step = 0
        with ExitStack() as stack:
            progress = stack.enter_context(tqdm(total=self.max_steps,
                desc=f"OneLeg {records[0]['rollout'] + 1}-{records[-1]['rollout'] + 1}/{self.n_rollouts}",
                unit='step', mininterval=1.0, miniters=1))
            writers = []
            if n_videos:
                writers, paths = self._open_videos(records[0]['rollout'], n_videos, video_dir)
                for writer, record, path in zip(writers, records, paths):
                    stack.callback(writer.close)
                    record['video'] = path
                self._write_frames(writers, obs, active)
            while active.any() and step < self.max_steps:
                if not actions:
                    prediction = policy.predict_action({key: value[:, None]
                        for key, value in obs.items()})['action_pred']
                    if prediction.shape != (self.n_envs, 16, 10) or not torch.isfinite(prediction).all():
                        raise ValueError('Policy must predict finite (B, 16, 10) actions')
                    actions.extend(prediction[:, :self.n_action_steps].unbind(1))
                try:
                    obs, reward, done, _ = env.step(actions.popleft())
                    success = env.is_success() | (reward.reshape(self.n_envs) > 0)
                except RuntimeError as error:
                    return step, error
                step += 1
                progress.update(1)
                self._write_frames(writers, obs, active)
                finished = active & (success | done.reshape(self.n_envs))
                for i in finished.nonzero().flatten().tolist():
                    status = 'success' if success[i] else (
                        'timeout' if step == self.max_steps else 'failure')
                    records[i].update(result=status, epi_len=step)
                active &= ~finished
        return step, None

    @torch.no_grad()
    def run(self, policy):
        if policy.n_obs_steps != 1 or policy.horizon != 16:
            raise ValueError('OneLeg runner requires a single-frame, horizon-16 base')
        device = torch.device(policy.device)
        index = device.index if device.index is not None else 0
        training = policy.training
        self.last_results = []
        env = None
        video_dir = None
        try:
            policy.eval()
            if self.record_video and self.n_vis > 0:
                video_dir = pathlib.Path(self.output_dir) / 'videos' / str(time.time_ns())
                video_dir.mkdir(parents=True, exist_ok=True)
            for offset in range(0, self.n_rollouts, self.n_envs):
                records = [
                    dict(rollout=offset + i, seed=self.seed + offset + i, env_index=i, result=None, epi_len=0)
                    for i in range(self.n_envs)
                ]
                step, error = 0, None
                try:
                    if env is None:
                        env = OneLegEnv(
                            num_envs=self.n_envs, max_env_steps=self.max_steps, render_size=self.render_size,
                            compute_device_id=index,
                            graphics_device_id=(index if self.graphics_device_id is None else self.graphics_device_id)
                        )
                    obs = env.reset(seed=self.seed + offset)
                except RuntimeError as exception:
                    error = exception
                else:
                    step, error = self._run_batch(env, policy, obs, records, video_dir)
                for record in records:
                    if record['result'] is None:
                        record.update(result='failure' if error else 'timeout', epi_len=step)
                        if error:
                            record['error'] = str(error)
                self.last_results.extend(records)
                if error:
                    logger.error('OneLeg simulation failed at rollout %d, step %d: %s', offset, step, error)
                    self._close_env(env)
                    env = None
        finally:
            self._close_env(env)
            policy.train(training)

        counts = {
            name: sum(record['result'] == name for record in self.last_results)
            for name in ('success', 'timeout', 'failure')
        }
        lengths = [
            record['epi_len']
            for record in self.last_results if record['result'] == 'success'
        ]

        return {
            **{f'test/{name}_rate': value / self.n_rollouts for name, value in counts.items()},
            **{f'test/{name}_count': value for name, value in counts.items()},
            'test/n_rollouts': self.n_rollouts,
            'test/success_mean_epi_len': float(np.mean(lengths)) if lengths else float('nan'),
        }
