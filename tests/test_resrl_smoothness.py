import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from zprl.policy.residue_policy import ResiduePolicy


class _ZeroQ(nn.Module):
    def __init__(self, num_qs):
        super().__init__()
        self.num_qs = num_qs

    def forward(self, obs, action):
        value = action.sum(dim=-1, keepdim=True) * 0.0
        return value.unsqueeze(0).expand(self.num_qs, -1, -1)


class _LinearActor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(action_dim))
        self.input_scale = nn.Parameter(torch.tensor(2.0))

    def _mean(self, actor_input):
        return self.offset + self.input_scale * actor_input[..., :self.offset.shape[0]]

    def get_action(self, actor_input):
        mean = self._mean(actor_input)
        return {
            'sample': mean,
            'mean': mean,
            'log_prob': mean.sum(dim=-1, keepdim=True) * 0.0,
        }

    def get_eval_action(self, actor_input):
        return self._mean(actor_input)


def _make_batch(obs_dim, base_naction):
    batch_size, action_dim = base_naction.shape
    return SimpleNamespace(
        observations=torch.zeros(batch_size, obs_dim),
        next_observations=torch.zeros(batch_size, obs_dim),
        actions=torch.cat([
            torch.zeros_like(base_naction),
            base_naction,
            torch.zeros_like(base_naction),
        ], dim=-1),
        rewards=torch.zeros(batch_size, 1),
        dones=torch.zeros(batch_size, 1),
    )


class TestResRLSmoothness(unittest.TestCase):
    def test_config_validation(self):
        kwargs = dict(obs_dim=3, action_dim=6, hidden_dim=8)
        invalid_kwargs = [
            dict(lambda_s=1.0, lambda_t=None, n_action_steps=3),
            dict(lambda_s=-1.0, lambda_t=1.0, n_action_steps=3),
            dict(lambda_s=1.0, lambda_t=1.0, sigma=-0.1, n_action_steps=3),
            dict(lambda_s=1.0, lambda_t=1.0, n_action_steps=1),
            dict(lambda_s=1.0, lambda_t=1.0, n_action_steps=4),
        ]
        for invalid in invalid_kwargs:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ResiduePolicy(**kwargs, **invalid)

    def test_disabled_path_matches_original_actor_loss(self):
        original = ResiduePolicy(
            obs_dim=3, action_dim=4, hidden_dim=8, auto_alpha=False)
        disabled = ResiduePolicy(
            obs_dim=3,
            action_dim=4,
            hidden_dim=8,
            auto_alpha=False,
            n_action_steps=2,
            lambda_s=None,
            lambda_t=None,
        )
        disabled.load_state_dict(original.state_dict())
        self.assertEqual(set(original.state_dict()), set(disabled.state_dict()))

        batch = _make_batch(3, torch.randn(5, 4))
        torch.manual_seed(123)
        original_loss, _ = original.compute_actor_loss(batch)
        torch.manual_seed(123)
        with patch.object(
                disabled.actor,
                'get_eval_action',
                wraps=disabled.actor.get_eval_action) as get_eval_action:
            disabled_loss, info = disabled.compute_actor_loss(batch)

        torch.testing.assert_close(disabled_loss, original_loss)
        get_eval_action.assert_not_called()
        self.assertEqual(info['spatial_smoothness_loss'], 0.0)
        self.assertEqual(info['temporal_smoothness_loss'], 0.0)

    def test_composite_smoothness_values_and_gradients(self):
        policy = ResiduePolicy(
            obs_dim=3,
            action_dim=3,
            actor_input='obs',
            hidden_dim=8,
            auto_alpha=False,
            res_scale=0.5,
            n_action_steps=3,
            lambda_s=3.0,
            lambda_t=4.0,
            sigma=0.1,
        )
        policy.actor = _LinearActor(action_dim=3)
        policy.qs = _ZeroQ(num_qs=policy.num_qs)
        batch = _make_batch(3, torch.tensor([[0.0, 2.0, 4.0]]))

        with patch('torch.randn_like', return_value=torch.ones(1, 3)):
            actor_loss, info = policy.compute_actor_loss(batch)

        expected_spatial = 0.5 * 3 * (0.5 * 2.0 * 0.1) ** 2
        expected_temporal = 0.5 * ((2.0 - 0.0) ** 2 + (4.0 - 2.0) ** 2) / 2
        expected_loss = 3.0 * expected_spatial + 4.0 * expected_temporal
        self.assertAlmostEqual(info['actor_rl_loss'], 0.0)
        self.assertAlmostEqual(info['spatial_smoothness_loss'], expected_spatial)
        self.assertAlmostEqual(info['temporal_smoothness_loss'], expected_temporal)
        self.assertAlmostEqual(actor_loss.item(), expected_loss, places=6)

        actor_loss.backward()
        self.assertIsNotNone(policy.actor.input_scale.grad)
        self.assertIsNotNone(policy.actor.offset.grad)
        self.assertNotEqual(policy.actor.input_scale.grad.item(), 0.0)
        self.assertGreater(policy.actor.offset.grad.abs().sum().item(), 0.0)

    def test_enabled_default_actor_input_is_finite_and_backward(self):
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            policy = ResiduePolicy(
                obs_dim=5,
                action_dim=6,
                hidden_dim=8,
                n_action_steps=3,
                lambda_s=0.5,
                lambda_t=0.5,
            )
            batch = _make_batch(5, torch.randn(4, 6))

            actor_loss, info = policy.compute_actor_loss(batch)
            self.assertTrue(torch.isfinite(actor_loss))
            self.assertGreaterEqual(info['spatial_smoothness_loss'], 0.0)
            self.assertGreaterEqual(info['temporal_smoothness_loss'], 0.0)
            actor_loss.backward()
            actor_grads = [
                parameter.grad
                for parameter in policy.actor.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(actor_grads)
            self.assertTrue(all(torch.all(torch.isfinite(grad)) for grad in actor_grads))

    def test_zero_weights_keep_raw_losses_only(self):
        policy = ResiduePolicy(
            obs_dim=3,
            action_dim=3,
            actor_input='obs',
            hidden_dim=8,
            auto_alpha=False,
            res_scale=0.5,
            n_action_steps=3,
            lambda_s=0.0,
            lambda_t=0.0,
            sigma=0.1,
        )
        policy.actor = _LinearActor(action_dim=3)
        policy.qs = _ZeroQ(num_qs=policy.num_qs)
        batch = _make_batch(3, torch.tensor([[0.0, 2.0, 4.0]]))

        with patch('torch.randn_like', return_value=torch.ones(1, 3)):
            actor_loss, info = policy.compute_actor_loss(batch)

        self.assertGreater(info['spatial_smoothness_loss'], 0.0)
        self.assertGreater(info['temporal_smoothness_loss'], 0.0)
        self.assertEqual(info['weighted_spatial_smoothness_loss'], 0.0)
        self.assertEqual(info['weighted_temporal_smoothness_loss'], 0.0)
        self.assertEqual(actor_loss.item(), info['actor_rl_loss'])


if __name__ == '__main__':
    unittest.main()
