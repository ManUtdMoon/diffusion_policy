import unittest

import torch

from zprl.policy.residue_policy import SumPolicy


class IdentityNormalizer:
    def unnormalize(self, x):
        return x


class DummyBasePolicy:
    def __init__(self, obs_emb, n_action_steps, action_dim):
        self.normalizer = {'action': IdentityNormalizer()}
        self.obs_emb = obs_emb
        self.naction = torch.zeros(
            obs_emb.shape[0], n_action_steps, action_dim)
        self.last_obs = None

    def eval(self):
        return self

    def predict_action(self, obs_dict):
        self.last_obs = obs_dict
        return {
            'obs_emb': self.obs_emb,
            'naction': self.naction,
        }


class DummyResiduePolicy:
    actor_input = 'obs'

    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.obs_dim = None
        self.last_input = None

    def predict_res_naction(self, obs_emb, argmax=False):
        self.last_input = obs_emb
        return obs_emb[:, -1:].repeat(1, self.action_dim)


class SumPolicySubtaskTest(unittest.TestCase):
    def make_policy(self, subtask_dim):
        obs_emb = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        base_policy = DummyBasePolicy(obs_emb, n_action_steps=2, action_dim=1)
        res_policy = DummyResiduePolicy(action_dim=2)
        res_policy.obs_dim = 4 + subtask_dim
        policy = SumPolicy(
            res_scale=1.0,
            base_obs_emb_dim=4,
            subtask_dim=subtask_dim,
            action_dim=1,
            n_action_steps=2,
            base_policy=base_policy,
            res_policy=res_policy,
        )
        return policy, base_policy, res_policy

    def test_checklist_only_enters_residual_state(self):
        policy, base_policy, res_policy = self.make_policy(subtask_dim=2)
        obs = {
            'state': torch.zeros(2, 2, 1),
            'completed_stage_mask': torch.tensor([
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ]),
        }

        result = policy.predict_action(obs)

        self.assertNotIn('completed_stage_mask', base_policy.last_obs)
        self.assertIn('completed_stage_mask', obs)
        expected = torch.cat([
            base_policy.obs_emb[:, -4:],
            obs['completed_stage_mask'][:, -1],
        ], dim=-1)
        torch.testing.assert_close(res_policy.last_input, expected)
        torch.testing.assert_close(
            result['action'][:, 0, 0], torch.tensor([0.0, 1.0]))

    def test_zero_subtask_dim_keeps_base_path(self):
        policy, base_policy, res_policy = self.make_policy(subtask_dim=0)
        obs = {'state': torch.zeros(2, 2, 1)}

        policy.predict_action(obs)

        self.assertIs(base_policy.last_obs, obs)
        torch.testing.assert_close(
            res_policy.last_input, base_policy.obs_emb[:, -4:])

    def test_train_action_requires_augmented_embedding(self):
        policy, base_policy, res_policy = self.make_policy(subtask_dim=2)
        obs_emb = torch.zeros(2, 6)

        policy.predict_train_action(base_policy.naction, obs_emb)

        torch.testing.assert_close(res_policy.last_input, obs_emb)
        with self.assertRaises(AssertionError):
            policy.predict_train_action(base_policy.naction, obs_emb[:, :-1])


if __name__ == '__main__':
    unittest.main()
