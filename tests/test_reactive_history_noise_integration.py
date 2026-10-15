import copy
import json
import os
import pathlib
import unittest

import av
import dill
import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from zprl.common.reactive_history import NoiseHistoryReplayBufferSamples
from zprl.model.vision.crop_randomizer import CROP_RANDOMIZER_TYPES
from zprl.policy.reactive_history_noise_policy import HistoryNoiseSumPolicy


ROOT = pathlib.Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('ZPRL_HISTORY_NOISE_GPU_SMOKE') == '1',
                     'Opt in to the real checkpoint/GPU/video integration check')
class HistoryNoiseGpuIntegrationTest(unittest.TestCase):
    def test_real_unet_gradients_inductor_and_runner_video(self):
        self.assertTrue(torch.cuda.is_available())
        device = torch.device('cuda:0')
        output = ROOT / '.doc/20260926-reactive_rl/r7-validation'
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = 'data/upload/offline/tool_hang/novib/checkpoints/latest.ckpt'
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'zprl/config')):
            cfg = compose(config_name='train_online_reactive_history_noise_robomimic_workspace', overrides=[
                'online_task=tool_hang_image_abs', f'online_task.base_ckpt={checkpoint}',
                'n_action_steps=4', 'noise_policy.history_length=3',
                'online_task.env_runner.n_train=0', 'online_task.env_runner.n_train_vis=0',
                'online_task.env_runner.n_test=1', 'online_task.env_runner.n_test_vis=1',
                'online_task.env_runner.n_envs=1', 'online_task.env_runner.max_steps=16',
            ])
        payload = torch.load(ROOT / checkpoint, map_location='cpu', pickle_module=dill)
        base_cfg = payload['cfg']
        self.assertEqual(base_cfg.task_name, cfg.task_name)
        base_cfg.policy.n_action_steps = cfg.n_action_steps
        base = hydra.utils.instantiate(base_cfg.policy)
        base.load_state_dict(payload['state_dicts']['ema_model'])
        base.to(device).requires_grad_(False).eval()
        for module in base.modules():
            if isinstance(module, CROP_RANDOMIZER_TYPES):
                module.force_random_crop = False
        do = base.n_obs_steps * base.obs_feature_dim + 1
        policy = hydra.utils.instantiate(cfg.noise_policy, obs_dim=do, action_dim=10, exec_dim=40).to(device)
        compiled = copy.deepcopy(policy)
        for module in (compiled.actor, compiled.qs, compiled.q_targets):
            module.forward = torch.compile(module.forward, fullgraph=True, dynamic=False)
        bs, h = 8, policy.history_length
        obs = torch.randn(bs, h, do, device=device)
        executed = torch.randn(bs, h, 40, device=device)
        valid = torch.ones(bs, h, device=device, dtype=torch.bool)
        valid[0, :-1] = False
        steps = torch.zeros(bs, h, device=device, dtype=torch.long)
        batch = NoiseHistoryReplayBufferSamples(
            obs, executed, valid, steps, obs + 1, executed + 1, valid, steps,
            torch.rand(bs, 10, device=device) * 2 - 1,
            torch.zeros(bs, 1, device=device), torch.zeros(bs, 1, device=device),
            torch.full((bs, 1), policy.gamma ** 2, device=device))
        losses = {}
        for method in ('compute_critic_loss', 'compute_actor_loss', 'compute_alpha_loss'):
            policy.zero_grad()
            compiled.zero_grad()
            torch.manual_seed(40)
            eager = getattr(policy, method)(batch)
            torch.manual_seed(40)
            actual = getattr(compiled, method)(batch)
            eager = eager[0] if isinstance(eager, tuple) else eager
            actual = actual[0] if isinstance(actual, tuple) else actual
            torch.testing.assert_close(actual, eager, atol=1e-4, rtol=1e-4)
            eager.backward()
            actual.backward()
            for left, right in zip(policy.parameters(), compiled.parameters()):
                if left.grad is None:
                    self.assertIsNone(right.grad)
                else:
                    self.assertTrue(torch.isfinite(right.grad).all())
                    torch.testing.assert_close(right.grad, left.grad, atol=2e-4, rtol=2e-3)
            self.assertTrue(all(parameter.grad is None for parameter in compiled.q_targets.parameters()))
            if method == 'compute_actor_loss':
                self.assertTrue(all(parameter.grad is None for parameter in compiled.qs.parameters()))
                self.assertTrue(all(cell.weight_ih.grad.abs().sum() > 0 for cell in compiled.actor.encoder.cells))
            losses[method] = actual.item()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in base.parameters()))
        self.assertFalse(base.training)

        combined = HistoryNoiseSumPolicy(
            noise_scale=cfg.training.noise_scale, base_obs_emb_dim=do - 1, subtask_dim=1,
            action_dim=10, n_action_steps=4, horizon=16, n_obs_steps=1,
            base_policy=base, noise_policy=policy).eval()
        runner = hydra.utils.instantiate(cfg.online_task.env_runner, output_dir=str(output))
        try:
            metrics = runner.run(combined)
        finally:
            runner.env.close()
        self.assertIsNone(combined.pending)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in base.parameters()))
        video = output / 'media/test_100000.mp4'
        frames = []
        with av.open(str(video)) as container:
            for frame in container.decode(video=0):
                pixels = frame.to_ndarray(format='rgb24')
                self.assertEqual(pixels.shape, (240, 240, 3))
                self.assertTrue(np.isfinite(pixels).all())
                frames.append(frame.pts)
        self.assertGreater(len(frames), 0)
        self.assertTrue(all(right > left for left, right in zip(frames, frames[1:])))
        report = {
            'checkpoint': checkpoint, 'base_class': type(base).__name__,
            'base_frozen_no_grad': True, 'compiled_backend': 'inductor',
            'eager_compiled_losses_and_gradients': 'passed', 'losses': losses,
            'video': str(video), 'decoded_frames': len(frames),
            'runner_metrics': {key: float(value) for key, value in metrics.items()
                               if isinstance(value, (int, float, np.number))},
            'note': 'Validation rollout of a fresh policy; this is not a trained performance result.',
        }
        (output / 'gpu-video-validation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    unittest.main()
