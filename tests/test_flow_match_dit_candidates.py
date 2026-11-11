import unittest

import torch

from zprl.model.common.normalizer import SingleFieldLinearNormalizer
from zprl.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from zprl.policy.flow_match_dit_image_policy import FlowMatchDitImagePolicy
from zprl.workspace.train_diffusion_image_accelerate_workspace import _validation_candidates


class ConditionVelocity(torch.nn.Module):
    def forward(self, x, t, cond):
        return cond.sum(-1)[:, None, None].expand_as(x)


class FlowMatchDitCandidatesTest(unittest.TestCase):
    def make_policy(self):
        shape_meta = {'obs': {'state': {'shape': [2]}}, 'action': {'shape': [2]}}
        encoder = MultiImageObsEncoder(shape_meta, torch.nn.Identity())
        policy = FlowMatchDitImagePolicy(
            shape_meta, encoder, horizon=8, n_action_steps=3, n_obs_steps=2,
            num_inference_steps=2, num_blocks=1, hidden_dim=8, time_dim=8,
            dim_feedforward=16, nhead=2, dropout=0)
        policy.model = ConditionVelocity()
        policy.normalizer['state'] = SingleFieldLinearNormalizer.create_identity()
        policy.normalizer['action'] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.full((2,), 2.), offset=torch.ones(2),
            input_stats_dict={key: torch.ones(2) for key in ('min', 'max', 'mean', 'std')})
        return policy.eval()

    def test_candidates_preserve_condition_order_noise_normalization_and_action_slice(self):
        policy = self.make_policy()
        condition = torch.arange(8.).reshape(2, 4)
        for count in (1, 3):
            with self.subTest(count=count):
                torch.manual_seed(5)
                noise = torch.randn(2 * count, 8, 2).reshape(2, count, 8, 2)
                expected = noise - condition.sum(-1)[:, None, None, None]
                torch.manual_seed(5)
                result = policy.conditional_predict_k(condition, count)
                self.assertEqual(result['action_pred'].shape, (2, count, 8, 2))
                self.assertEqual(result['action'].shape, (2, count, 3, 2))
                torch.testing.assert_close(result['naction_pred'], expected)
                torch.testing.assert_close(result['action_pred'], (expected - 1) / 2)
                torch.testing.assert_close(result['naction'], expected[:, :, 1:4])
                torch.testing.assert_close(result['action'], (expected[:, :, 1:4] - 1) / 2)
                self.assertIs(result['obs_emb'], condition)
        torch.manual_seed(6)
        single = policy.conditional_predict_k(condition, 1)
        torch.manual_seed(6)
        original = policy.conditional_predict(condition)
        for key in ('action', 'action_pred', 'naction', 'naction_pred'):
            torch.testing.assert_close(single[key][:, 0], original[key], rtol=0, atol=0)

    def test_condition_requires_full_observation_embedding(self):
        policy = self.make_policy()
        with self.assertRaises(AssertionError):
            policy.conditional_predict_k(torch.zeros(2, 2), 3)

    def test_cov_rel_validation_handles_partial_decode_batch(self):
        policy = self.make_policy()
        batch = {'obs': {'state': torch.randn(3, 3, 2)}, 'action': torch.randn(3, 8, 2)}
        with torch.no_grad():
            first, quantiles = _validation_candidates(policy, batch, decode_batch_size=2, num_candidates=3)
        self.assertEqual(first.shape, batch['action'].shape)
        self.assertEqual(quantiles.shape[0], 3)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(quantiles).all())


if __name__ == '__main__':
    unittest.main()
