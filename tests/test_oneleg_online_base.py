import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from zprl.common.oneleg_online import load_oneleg_base


class OneLegOnlineBaseValidationTest(unittest.TestCase):
    def test_reject_incompatible_checkpoint_before_instantiation(self):
        cfg = OmegaConf.create(dict(task_name='oneleg_image', task={'name': 'oneleg_image'},
            horizon=16, n_obs_steps=1,
            policy={'_target_': 'zprl.policy.flow_match_unet_image_policy.FlowMatchUnetImagePolicy'},
            shape_meta={'obs': {'color_image1': {'shape': [3, 240, 320]},
                                'color_image2': {'shape': [3, 240, 320]},
                                'robot_state': {'shape': [16]}}, 'action': {'shape': [10]}}))
        cases = [('task_name', 'square_image'), ('task.name', 'square_image'),
                 ('n_obs_steps', 2), ('horizon', 8),
                 ('shape_meta.action.shape', [8]), ('shape_meta.obs.robot_state.shape', [17])]
        for key, value in cases:
            with self.subTest(key=key):
                invalid = OmegaConf.create(OmegaConf.to_container(cfg))
                OmegaConf.update(invalid, key, value)
                with tempfile.NamedTemporaryFile() as file, \
                     patch('zprl.common.oneleg_online.torch.load', return_value={'cfg': invalid}), \
                     patch('zprl.common.oneleg_online.hydra.utils.instantiate') as instantiate:
                    with self.assertRaises(ValueError):
                        load_oneleg_base(file.name)
                    instantiate.assert_not_called()


LIVE_CHECK = r'''
import isaacgym
import hashlib
import sys
import torch
from zprl.common.oneleg_online import load_oneleg_base
from zprl.policy.reactive_history_policy import HistoryResiduePolicy, HistorySumPolicy
from zprl.policy.reactive_history_noise_policy import HistoryNoisePolicy, HistoryNoiseSumPolicy

base, cfg = load_oneleg_base(sys.argv[1])
assert base.device.type == 'cpu'
assert base.num_inference_steps == 5
assert all(not module.training for module in base.modules())
assert not any(parameter.requires_grad for parameter in base.parameters())
print('CPU_EMA_LOAD_OK', base.obs_feature_dim, flush=True)

def fingerprint():
    digest = hashlib.sha256()
    for key, value in base.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

original = fingerprint()
if sys.argv[2] == 'cpu':
    obs = {key: torch.zeros(2, 1, *meta.shape) for key, meta in cfg.shape_meta.obs.items()}
else:
    from zprl.env.furniturebench.oneleg_env import OneLegEnv
    env = OneLegEnv(num_envs=2, compute_device_id=0, graphics_device_id=0)
    base = base.to(env.device)
    obs = {key: value[:, None] for key, value in env.reset(seed=10000).items()}
try:
    with torch.no_grad():
        embedding = base.encode_obs(obs)
        torch.testing.assert_close(embedding, base.encode_obs(obs), rtol=0, atol=0)
        assert embedding.shape == (2, base.n_obs_steps * base.obs_feature_dim)
        state = torch.get_rng_state() if base.device.type == 'cpu' else torch.cuda.get_rng_state(base.device)
        noise = torch.randn(2, 16, 10, device=base.device, dtype=base.dtype)
        original_noise = noise.clone()
        if base.device.type == 'cpu':
            torch.set_rng_state(state)
        else:
            torch.cuda.set_rng_state(state, base.device)
        ordinary = base.conditional_predict(embedding)
        controlled = base.conditional_predict_from_noise(embedding, noise)
        repeated = base.conditional_predict_from_noise(embedding, noise)
        for result in (ordinary, controlled):
            assert result['obs_emb'].shape == embedding.shape
            for field, shape in [('action_pred', (2, 16, 10)), ('naction_pred', (2, 16, 10)),
                                 ('action', (2, 8, 10)), ('naction', (2, 8, 10))]:
                assert result[field].shape == shape, (field, result[field].shape)
                assert torch.isfinite(result[field]).all()
                assert not result[field].requires_grad
                torch.testing.assert_close(result[field], repeated[field], rtol=0, atol=0)
            torch.testing.assert_close(base.normalizer['action'].normalize(result['action_pred']),
                result['naction_pred'], rtol=1e-4, atol=1e-5)
        assert torch.equal(noise, original_noise)
        print('ENCODE_AND_BOTH_DECODERS_OK', sys.argv[2], flush=True)
        if sys.argv[2] != 'cpu':
            dim = embedding.shape[-1]
            common = dict(obs_dim=dim, action_dim=80, history_length=1, sub_ratio=1)
            res = HistoryResiduePolicy(**common).to(base.device)
            summed = HistorySumPolicy(res_scale=0.05, base_obs_emb_dim=dim, subtask_dim=0,
                action_dim=10, n_action_steps=8, n_rl_steps=8, base_policy=base, res_policy=res)
            noise_policy = HistoryNoisePolicy(obs_dim=dim, action_dim=10, exec_dim=80,
                history_length=1, sub_ratio=1).to(base.device)
            noisy = HistoryNoiseSumPolicy(noise_scale=1., base_obs_emb_dim=dim, subtask_dim=0,
                action_dim=10, n_action_steps=8, horizon=16, n_obs_steps=1,
                base_policy=base, noise_policy=noise_policy, n_noise_steps=1)
            for composed in (summed, noisy):
                composed.train()
                assert not base.training
                composed.reset(obs)
                action = composed.predict_action(obs)['action']
                assert action.shape == (2, 8, 10)
                assert torch.isfinite(action).all()
                for step in range(8):
                    next_obs, reward, done, info = env.step(action[:, step])
                    assert all(torch.isfinite(value).all() for value in next_obs.values())
                obs = {key: value[:, None] for key, value in next_obs.items()}
            print('TWO_POLICY_GPU_ACTION_SMOKE_OK', flush=True)
    assert fingerprint() == original, 'Frozen EMA/normalizer changed'
    assert not any(parameter.requires_grad for parameter in base.parameters())
    print('FROZEN_BASE_UNCHANGED_OK', flush=True)
finally:
    if sys.argv[2] != 'cpu':
        env.close()
'''


@unittest.skipUnless(os.environ.get('ONELEG_ONLINE_BASE_CHECKPOINT'),
    'Opt in with ONELEG_ONLINE_BASE_CHECKPOINT; live checks run in fresh processes')
class OneLegOnlineBaseIntegrationTest(unittest.TestCase):
    def test_real_ema_on_cpu(self):
        self.run_check('cpu')

    @unittest.skipUnless(os.environ.get('ONELEG_ONLINE_GPU_SMOKE') == '1', 'Opt in to GPU simulation')
    def test_real_ema_and_two_policies_on_gpu(self):
        self.run_check('gpu')

    def run_check(self, device):
        result = subprocess.run([sys.executable, '-c', LIVE_CHECK,
            os.environ['ONELEG_ONLINE_BASE_CHECKPOINT'], device],
            cwd=pathlib.Path(__file__).resolve().parents[1],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='1,0,2,3', OMP_NUM_THREADS='2'),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('ENCODE_AND_BOTH_DECODERS_OK', result.stdout)
        self.assertIn('FROZEN_BASE_UNCHANGED_OK', result.stdout)
        if device == 'gpu':
            self.assertIn('TWO_POLICY_GPU_ACTION_SMOKE_OK', result.stdout)
        print(result.stdout, flush=True)


if __name__ == '__main__':
    unittest.main()
