import ast
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from zprl.common.online_util import get_crop_randomizers, prepare_base_policy_config
from zprl.model.online import BatchedLayerNorm, BatchedLinear, BatchedSoftQNet
from zprl.model.vision.crop_randomizer import CropRandomizerV2, CropRandomizerV3
from zprl.policy.noise_policy import NoisePolicy, SumPolicy


class _FixedQ(nn.Module):
    def __init__(self, num_qs, value):
        super().__init__()
        self.num_qs = num_qs
        self.value = value

    def forward(self, obs, action):
        return torch.full(
            (self.num_qs, obs.shape[0], 1), self.value,
            device=obs.device, dtype=obs.dtype)


class _BasePolicyStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.horizon = 4
        self.action_dim = 2
        self.normalizer = None
        self.last_noise = None

    def encode_obs(self, obs_dict):
        return obs_dict['obs']

    def conditional_predict_from_noise(self, obs_emb, noise):
        self.last_noise = noise
        return {'action': noise[:, :2]}


class _NoisePolicyStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_dim = 2
        self.argmax_calls = []

    def predict_noise(self, obs_emb, argmax=False):
        self.argmax_calls.append(argmax)
        return torch.zeros(obs_emb.shape[0], self.action_dim)


class OnlineFeatureMigrationTest(unittest.TestCase):
    def test_legacy_base_config_patch_only_applies_allowed_overrides(self):
        cfg = OmegaConf.create({
            'task_name': 'can_image_abs',
            'policy': {
                '_target_': 'diffusion_policy.policy.fake.FakePolicy',
                'n_action_steps': 4,
                'num_inference_steps': 1,
                'hidden_dim': 123,
            },
            'untouched': {'seed': 7},
        })

        result = prepare_base_policy_config(cfg, 8, 20)

        self.assertEqual(result.policy._target_, 'zprl.policy.fake.FakePolicy')
        self.assertEqual(result.policy.n_action_steps, 8)
        self.assertEqual(result.policy.num_inference_steps, 20)
        self.assertEqual(result.policy.hidden_dim, 123)
        self.assertEqual(result.untouched.seed, 7)
        self.assertEqual(result.task_name, 'can_image_abs')

    def test_crop_randomizer_capability_detects_v2_and_v3(self):
        v2 = CropRandomizerV2((3, 6, 6), 4, 4)
        v3 = CropRandomizerV3((3, 6, 6), 4, 4)
        policy = nn.Sequential(nn.Identity(), v2, nn.Sequential(v3))

        randomizers = get_crop_randomizers(policy)

        self.assertEqual(randomizers, [v2, v3])
        for randomizer in randomizers:
            randomizer.force_random_crop = True
        self.assertTrue(v2.force_random_crop)
        self.assertTrue(v3.force_random_crop)

    def test_batched_layer_norm_forward_backward(self):
        layer = BatchedLayerNorm(3, 5)
        with torch.no_grad():
            layer.weight[1].fill_(2.0)
            layer.bias[1].fill_(1.0)
        inputs = torch.randn(3, 4, 5, requires_grad=True)

        output = layer(inputs)
        normalized = F.layer_norm(inputs, (5,))

        self.assertEqual(output.shape, inputs.shape)
        torch.testing.assert_close(output[0], normalized[0])
        torch.testing.assert_close(output[1], normalized[1] * 2.0 + 1.0)
        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)

    def test_default_batched_q_contract_is_unchanged(self):
        q_net = BatchedSoftQNet(obs_dim=3, action_dim=2, num_qs=2, hidden_dim=4)
        expected_types = [
            BatchedLinear, nn.GELU, BatchedLinear, nn.GELU,
            BatchedLinear, nn.GELU, BatchedLinear,
        ]
        expected_keys = {
            'net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias',
            'net.4.weight', 'net.4.bias', 'net.6.weight', 'net.6.bias',
        }

        self.assertEqual([type(module) for module in q_net.net], expected_types)
        self.assertEqual(set(q_net.state_dict()), expected_keys)
        self.assertFalse(any(isinstance(module, BatchedLayerNorm) for module in q_net.modules()))
        output = q_net(torch.randn(5, 3), torch.randn(5, 2))
        self.assertEqual(output.shape, (2, 5, 1))
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))

    def _make_noise_policy(self, q_entropy=True):
        return NoisePolicy(
            obs_dim=3,
            action_dim=2,
            hidden_dim=8,
            gamma=0.9,
            init_alpha=0.25,
            auto_alpha=False,
            target_entropy=0.0,
            num_qs=2,
            num_subset=2,
            q_entropy=q_entropy,
        )

    @staticmethod
    def _fix_noise_policy(policy):
        policy.q_targets = _FixedQ(2, 0.5)
        policy.qs = _FixedQ(2, 0.25)
        policy._sample_log_prob = lambda obs: (
            torch.zeros(obs.shape[0], 2),
            torch.full((obs.shape[0], 1), -2.0),
            torch.zeros(obs.shape[0], 2),
        )

    def test_q_entropy_only_changes_critic_target(self):
        soft = self._make_noise_policy(q_entropy=True)
        hard = self._make_noise_policy(q_entropy=False)
        self.assertEqual(set(soft.state_dict()), set(hard.state_dict()))
        self._fix_noise_policy(soft)
        self._fix_noise_policy(hard)
        batch = SimpleNamespace(
            observations=torch.zeros(4, 3),
            next_observations=torch.zeros(4, 3),
            actions=torch.zeros(4, 2),
            rewards=torch.zeros(4, 1),
            dones=torch.zeros(4, 1),
        )

        _, soft_info = soft.compute_critic_loss(batch)
        _, hard_info = hard.compute_critic_loss(batch)
        soft_actor, _ = soft.compute_actor_loss(batch)
        hard_actor, _ = hard.compute_actor_loss(batch)
        soft_alpha = soft.compute_alpha_loss(batch)
        hard_alpha = hard.compute_alpha_loss(batch)

        self.assertAlmostEqual(soft_info['q_target'], 0.9)
        self.assertAlmostEqual(hard_info['q_target'], 0.45)
        torch.testing.assert_close(soft_actor, hard_actor)
        torch.testing.assert_close(soft_alpha, hard_alpha)

    def test_q_entropy_missing_config_falls_back_true_and_checkpoint_loads(self):
        original = self._make_noise_policy()
        self.assertTrue(original.q_entropy)
        cfg = OmegaConf.create({
            '_target_': 'zprl.policy.noise_policy.NoisePolicy',
            'obs_dim': 3,
            'action_dim': 2,
            'hidden_dim': 8,
            'num_qs': 2,
            'num_subset': 2,
        })
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / 'noise.ckpt'
            with path.open('wb') as file:
                torch.save(
                    {'cfg': cfg, 'noise_policy': original.state_dict()},
                    file, pickle_module=dill)
            with path.open('rb') as file:
                payload = torch.load(file, pickle_module=dill)

        restored = NoisePolicy(
            obs_dim=3, action_dim=2, hidden_dim=8, num_qs=2, num_subset=2)
        restored.load_state_dict(payload['noise_policy'], strict=True)
        self.assertTrue(restored.q_entropy)
        self.assertNotIn('q_entropy', payload['cfg'])

    def test_dsrl_fake_batch_losses_are_finite_and_backward_for_multiple_seeds(self):
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            policy = self._make_noise_policy()
            batch = SimpleNamespace(
                observations=torch.randn(4, 3),
                next_observations=torch.randn(4, 3),
                actions=torch.rand(4, 2) * 2.0 - 1.0,
                rewards=torch.rand(4, 1),
                dones=torch.zeros(4, 1),
            )

            critic_loss, _ = policy.compute_critic_loss(batch)
            self.assertTrue(torch.isfinite(critic_loss))
            critic_loss.backward()
            self.assertTrue(any(
                parameter.grad is not None
                for parameter in policy.qs.parameters()))

            policy.zero_grad(set_to_none=True)
            actor_loss, _ = policy.compute_actor_loss(batch)
            self.assertTrue(torch.isfinite(actor_loss))
            actor_loss.backward()
            self.assertTrue(any(
                parameter.grad is not None
                for parameter in policy.actor.parameters()))

            policy.zero_grad(set_to_none=True)
            alpha_loss = policy.compute_alpha_loss(batch)
            self.assertTrue(torch.isfinite(alpha_loss))
            alpha_loss.backward()
            self.assertIsNotNone(policy.log_alpha.grad)

    def test_warmup_stores_the_executed_normalized_noise(self):
        base_policy = _BasePolicyStub()
        noise_policy = _NoisePolicyStub()
        sum_policy = SumPolicy(
            noise_scale=1.5,
            n_noise_steps=1,
            base_policy=base_policy,
            noise_policy=noise_policy,
        )
        sampled_noise = torch.tensor([[-2.0, 0.75]])

        with patch('torch.randn_like', return_value=sampled_noise):
            result = sum_policy.predict_train_action(torch.zeros(1, 3), perturb=False)

        expected_normalized = torch.tensor([[-1.0, 0.5]])
        torch.testing.assert_close(result['nnoise'], expected_normalized)
        torch.testing.assert_close(
            base_policy.last_noise,
            (expected_normalized * 1.5).reshape(1, 1, 2).repeat(1, 4, 1))

        sum_policy.predict_action({'obs': torch.zeros(1, 3)})
        self.assertFalse(noise_policy.argmax_calls[-1])

    def test_dsrl_workspace_keeps_active_eval_and_checkpoint(self):
        workspace_path = pathlib.Path(__file__).parents[1] / (
            'zprl/workspace/train_online_noise_robomimic_workspace.py')
        tree = ast.parse(workspace_path.read_text())
        calls = list(ast.walk(tree))
        eval_calls = [
            node for node in calls
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'eval_env_runner'
            and node.func.attr == 'run'
        ]
        save_calls = [
            node for node in calls
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'torch'
            and node.func.attr == 'save'
        ]

        self.assertGreaterEqual(len(eval_calls), 2)
        self.assertEqual(len(save_calls), 1)

    def test_online_workspaces_and_configs_have_no_q_pretrain(self):
        root = pathlib.Path(__file__).parents[1]
        paths = list((root / 'zprl/workspace').glob('train_online*_workspace.py'))
        paths += list((root / 'zprl/config').glob('train_online*_workspace.yaml'))
        self.assertTrue(paths)
        for path in paths:
            source = path.read_text()
            self.assertNotIn('q_pretrain_steps', source, str(path))
            self.assertNotIn('Q pre-training', source, str(path))


if __name__ == '__main__':
    unittest.main()
