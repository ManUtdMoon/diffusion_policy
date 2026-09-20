import copy
import json
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import dill
import hydra
import torch
from omegaconf import OmegaConf

from soe.eval_soe import episode_results, evaluate
from soe.soe_util import SoeBasePolicy, load_soe_policy
from zprl.common.normalize_util import get_image_range_normalizer


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TinyObsEncoder(torch.nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.proj = torch.nn.Linear(state_dim, 4)
        self.crop = torch.nn.Identity()
        self.crop.force_random_crop = True

    def output_shape(self):
        return (4,)

    def forward(self, obs):
        return self.proj(obs['agent_pos'])


def make_checkpoint(path, task='metaworld_box-close', use_ema=True):
    task_cfg = OmegaConf.load(ROOT / 'zprl/config/task' / f'{task}.yaml')
    state_dim = task_cfg.shape_meta.obs.agent_pos.shape[0]
    cfg = OmegaConf.create({
        '_target_': 'unused.Workspace',
        'task': task_cfg,
        'task_name': '${task.name}',
        'shape_meta': '${task.shape_meta}',
        'n_obs_steps': 2,
        'n_action_steps': 2,
        'horizon': 8,
        'training': {'use_ema': use_ema, 'seed': 7, 'device': 'cuda:3'},
        'policy': {
            '_target_': 'zprl.policy.flow_match_vib_unet_image_policy.FlowMatchVibUnetImagePolicy',
            'shape_meta': '${shape_meta}',
            'obs_encoder': {
                '_target_': f'{__name__}.TinyObsEncoder', 'state_dim': state_dim},
            'noise_scheduler': {
                '_target_': 'diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler',
                'num_train_timesteps': 10},
            'horizon': '${horizon}',
            'n_obs_steps': '${n_obs_steps}',
            'n_action_steps': '${n_action_steps}',
            'num_inference_steps': 2,
            'diffusion_step_embed_dim': 8,
            'down_dims': [8, 16],
            'n_groups': 4,
            'vib_latent_dim': 2,
            'vib_hidden_dim': 8,
        },
    })
    policy = hydra.utils.instantiate(cfg.policy)
    policy.normalizer.fit({
        'agent_pos': torch.randn(16, state_dim),
        'action': torch.randn(16, task_cfg.shape_meta.action.shape[0]),
    })
    policy.normalizer['image'] = get_image_range_normalizer()
    ema_state = copy.deepcopy(policy.state_dict())
    model_state = copy.deepcopy(ema_state)
    model_state['obs_encoder.proj.bias'].add_(1)
    payload = {'cfg': cfg, 'state_dicts': {
        'model': model_state, 'ema_model': ema_state}, 'pickles': {}}
    torch.save(payload, path, pickle_module=dill)
    return payload


class SoeEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.checkpoint = self.root / 'policy.ckpt'
        self.payload = make_checkpoint(self.checkpoint)

    def test_checkpoint_weights_normalizer_and_frozen_base_path(self):
        policy, cfg, weights = load_soe_policy(
            self.checkpoint, 'metaworld_box-close')
        self.assertEqual(weights, 'ema_model')
        self.assertFalse(policy.training)
        self.assertFalse(any(p.requires_grad for p in policy.parameters()))
        self.assertFalse(policy.obs_encoder.crop.force_random_crop)
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, self.payload['state_dicts'][weights][key])
        obs = {'image': torch.rand(2, 2, 3, 84, 84),
               'agent_pos': torch.rand(2, 2, 9)}
        torch.manual_seed(19)
        expected = policy.conditional_predict(policy.encode_obs(obs))
        with patch.object(policy.vib_encoder, 'forward', side_effect=AssertionError('VIB called')), \
                patch.object(policy.vib_decoder, 'forward', side_effect=AssertionError('VIB called')), \
                patch.object(policy, 'predict_action', side_effect=AssertionError('default path called')):
            torch.manual_seed(19)
            actual = SoeBasePolicy(policy).predict_action(obs)
        self.assertEqual(actual['action'].shape, (2, 2, 4))
        self.assertFalse(actual['action'].requires_grad)
        torch.testing.assert_close(actual['action'], expected['action'])

    def test_non_ema_and_legacy_hammer_checkpoint(self):
        payload = make_checkpoint(self.checkpoint, 'adroit_hammer', use_ema=False)
        payload['cfg'] = OmegaConf.create(OmegaConf.to_yaml(payload['cfg']).replace(
            'zprl.policy.', 'diffusion_policy.policy.'))
        torch.save(payload, self.checkpoint, pickle_module=dill)
        policy, cfg, weights = load_soe_policy(
            self.checkpoint, 'adroit_hammer', n_action_steps=3, num_inference_steps=4)
        self.assertEqual(weights, 'model')
        self.assertEqual((policy.n_action_steps, policy.num_inference_steps), (3, 4))
        torch.testing.assert_close(policy.obs_encoder.proj.bias,
            payload['state_dicts']['model']['obs_encoder.proj.bias'])

    def test_bad_task_shape_and_missing_ema_fail(self):
        with self.assertRaisesRegex(ValueError, 'does not match'):
            load_soe_policy(self.checkpoint, 'adroit_hammer')
        payload = copy.deepcopy(self.payload)
        payload['cfg'].policy.shape_meta = {'action': {'shape': [99]}}
        torch.save(payload, self.checkpoint, pickle_module=dill)
        with self.assertRaisesRegex(ValueError, 'shapes'):
            load_soe_policy(self.checkpoint, 'metaworld_box-close')
        del self.payload['state_dicts']['ema_model']
        torch.save(self.payload, self.checkpoint, pickle_module=dill)
        with self.assertRaisesRegex(ValueError, 'ema_model'):
            load_soe_policy(self.checkpoint, 'metaworld_box-close')

    def test_ae_and_invalid_chunk_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'horizon'):
            load_soe_policy(self.checkpoint, 'metaworld_box-close', n_action_steps=8)
        self.payload['cfg'].policy._target_ = (
            'zprl.policy.flow_match_ae_unet_image_policy.FlowMatchAeUnetImagePolicy')
        torch.save(self.payload, self.checkpoint, pickle_module=dill)
        with self.assertRaisesRegex(ValueError, 'VIB encoder|FlowMatchVib'):
            load_soe_policy(self.checkpoint, 'metaworld_box-close')

    def test_task_success_rules(self):
        hammer = episode_results('adroit_hammer', {
            'test/n_goal_10000': 49, 'test/n_goal_10001': 50}, [10000, 10001], 50)
        box = episode_results('metaworld_box-close', {
            'test/reward_0': 1, 'test/reward_1': 0}, [10000, 10001])
        self.assertEqual([e['success'] for e in hammer], [False, True])
        self.assertEqual([e['success'] for e in box], [True, False])

    def test_default_directory_metadata_and_runner_cleanup(self):
        loaded = load_soe_policy(self.checkpoint, 'metaworld_box-close')
        runner = unittest.mock.Mock()
        runner.run.return_value = {'test/reward_0': 1, 'test/reward_1': 0}
        old_cwd = os.getcwd()
        try:
            os.chdir(self.root)
            with patch('soe.eval_soe.load_soe_policy', return_value=loaded), \
                    patch('soe.eval_soe.hydra.utils.instantiate', return_value=runner) as instantiate:
                summary = evaluate(self.checkpoint, 'metaworld_box-close', 3, 0,
                    num_episodes=2, n_envs=1, device='cpu', policy_seed=5)
        finally:
            os.chdir(old_cwd)
        runner.close.assert_called_once()
        runner_cfg = instantiate.call_args.args[0]
        self.assertEqual(runner_cfg.n_obs_steps, 2)
        self.assertEqual(runner_cfg.n_action_steps, 2)
        self.assertEqual(runner_cfg.render_device_id, 0)
        self.assertEqual(runner_cfg.n_epi_vis, 0)
        output = self.root / 'data/soe/metaworld_box-close/seed_3/round_0/eval'
        metadata = json.loads((output / 'eval_config.json').read_text())
        episodes = [json.loads(line) for line in (output / 'eval_episodes.jsonl').read_text().splitlines()]
        self.assertEqual(metadata['eval_seeds'], [10000, 10001])
        self.assertEqual(metadata['policy_seed'], 5)
        self.assertFalse(metadata['vib_enabled'])
        self.assertNotIn('online_episodes', metadata)
        self.assertEqual(summary['eval_success_rate'], 0.5)
        self.assertEqual(len(episodes), 2)
        with self.assertRaisesRegex(ValueError, 'not empty'):
            evaluate(self.checkpoint, 'metaworld_box-close', 3, 0, output=output)

    def test_runner_is_closed_on_failure(self):
        loaded = load_soe_policy(self.checkpoint, 'metaworld_box-close')
        runner = unittest.mock.Mock()
        runner.run.side_effect = RuntimeError('rollout failed')
        output = self.root / 'failed_eval'
        with patch('soe.eval_soe.load_soe_policy', return_value=loaded), \
                patch('soe.eval_soe.hydra.utils.instantiate', return_value=runner), \
                self.assertRaisesRegex(RuntimeError, 'rollout failed'):
            evaluate(self.checkpoint, 'metaworld_box-close', 0, 0,
                output=output, device='cpu')
        runner.close.assert_called_once()
        self.assertFalse((output / 'eval_summary.json').exists())

    def test_invalid_episode_batch_fails_before_checkpoint_loading(self):
        with self.assertRaisesRegex(ValueError, 'divisible'):
            evaluate('missing.ckpt', 'metaworld_box-close', 0, 0,
                num_episodes=3, n_envs=2)


if __name__ == '__main__':
    unittest.main()
