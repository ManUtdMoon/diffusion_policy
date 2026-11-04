import io
import pathlib
import unittest
from unittest.mock import patch

import dill
import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from zprl.model.online import Actor
from zprl.policy.residue_policy import ResiduePolicy, SumPolicy
from tests.test_sum_policy_subtask import DummyBasePolicy


class ActionQ(torch.nn.Module):
    def forward(self, obs, action):
        # Different batch elements prefer opposite actions. Head 0 alone prefers
        # the extremes, while the subset minimum prefers the middle candidate.
        x = obs[:, :1] * action[:, :1]
        return torch.stack([x, 0.5 - x, -100 * torch.ones_like(x)])


class ResidueSelectionTest(unittest.TestCase):
    def make_policy(self, **kwargs):
        config = dict(obs_dim=2, action_dim=2, hidden_dim=8, num_qs=3,
                      num_subset=2, q_ent=False, res_scale=0.5,
                      num_res_candidates=3, select_rollout=True, select_backup=True)
        config.update(kwargs)
        return ResiduePolicy(**config)

    def make_batch(self):
        return ReplayBufferSamples(
            observations=torch.tensor([[3., 4.], [5., 6.]]),
            next_observations=torch.tensor([[1., 10.], [-1., 20.]]),
            actions=torch.tensor([[0.2, 0.4, 2., 3., 0., 0.],
                                  [-0.2, -0.4, 4., 5., 0., 0.]]),
            rewards=torch.tensor([[-0.2], [2.]]),
            dones=torch.tensor([[0.], [1.]]),
            discounts=torch.tensor([[0.5], [0.25]]))

    def test_candidates_use_one_forward_independent_noise_and_correct_axes(self):
        actor = Actor(3, 2, hidden_dim=8)
        mean = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
        logstd = torch.tensor([[-1., -2.], [-3., -4.]])
        noise = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 10
        with patch.object(actor, 'forward', return_value=(mean, logstd)) as forward, \
             patch('torch.distributions.normal._standard_normal', return_value=noise) as sample:
            result = actor.sample_candidates(torch.zeros(2, 3), 3)
        forward.assert_called_once()
        self.assertEqual(sample.call_args.args[0], (3, 2, 2))
        torch.testing.assert_close(result, (mean + logstd.exp() * noise).tanh().permute(1, 0, 2))
        result.sum().backward()
        self.assertGreater(mean.grad.abs().sum(), 0)
        x = torch.randn(2, 3)
        torch.manual_seed(7)
        single = actor.sample_candidates(x, 1)
        torch.manual_seed(7)
        torch.testing.assert_close(single[:, 0], actor.get_action(x)['sample'])
        self.assertEqual(single.shape, (2, 1, 2))

    def test_shared_target_subset_min_batch_selection_and_unscaled_output(self):
        policy = self.make_policy()
        policy.q_targets = ActionQ()
        batch = self.make_batch()
        candidates = torch.tensor([[[-1., 0.1], [0.5, 0.2], [1., 0.3]],
                                   [[-1., 0.4], [1., 0.6], [-0.5, 0.5]]], requires_grad=True)
        with patch.object(policy.actor, 'sample_candidates', return_value=candidates) as sample, \
             patch.object(policy.q_targets, 'forward', wraps=policy.q_targets.forward) as target, \
             patch.object(policy.qs, 'forward', side_effect=AssertionError('online Q used')), \
             patch('torch.randperm', return_value=torch.tensor([0, 1, 2])) as subset:
            result, score = policy._select_res_naction(batch.next_observations, torch.zeros(2, 2))
        torch.testing.assert_close(result, torch.stack([candidates[0, 1], candidates[1, 2]]))
        torch.testing.assert_close(score, torch.full((2, 1), 0.25))
        torch.testing.assert_close(sample.call_args.args[0][:, :2], batch.next_observations)
        torch.testing.assert_close(target.call_args.args[0], batch.next_observations.repeat_interleave(3, 0))
        torch.testing.assert_close(target.call_args.args[1], (0.5 * candidates).reshape(6, 2))
        subset.assert_called_once()
        self.assertFalse(result.requires_grad)
        self.assertFalse(score.requires_grad)

    def test_scoring_uses_base_anchor_without_clipping(self):
        policy = self.make_policy(num_res_candidates=1)
        policy.q_targets = ActionQ()
        obs = torch.ones(2, 2)
        base = torch.tensor([[2., 3.], [-4., -5.]])
        candidates = torch.tensor([[[0.2, 0.4]], [[-0.2, -0.4]]])
        with patch.object(policy.actor, 'sample_candidates', return_value=candidates) as sample, \
             patch.object(policy.q_targets, 'forward', wraps=policy.q_targets.forward) as target:
            selected, _ = policy._select_res_naction(obs, base)
        torch.testing.assert_close(sample.call_args.args[0], torch.cat([obs, base], -1))
        torch.testing.assert_close(target.call_args.args[1], torch.tensor([[2.1, 3.2], [-4.1, -5.2]]))
        torch.testing.assert_close(selected, candidates[:, 0])

    def test_zero_candidate_wins_and_ties_prefer_base(self):
        policy = self.make_policy(include_base_candidate=True)
        policy.q_targets = ActionQ()
        obs = torch.tensor([[1., 0.], [-1., 0.]])
        candidates = torch.ones(2, 3, 2)
        with patch.object(policy.actor, 'sample_candidates', return_value=candidates), \
             patch('torch.randperm', return_value=torch.tensor([0, 1, 2])):
            selected, _ = policy._select_res_naction(obs, torch.zeros(2, 2))
            torch.testing.assert_close(selected, torch.zeros(2, 2))
            # All candidate scores equal when the observation multiplier is zero.
            selected, _ = policy._select_res_naction(torch.zeros_like(obs), torch.zeros(2, 2))
            torch.testing.assert_close(selected, torch.zeros(2, 2))

    def test_backup_reuses_score_clips_complete_target_and_regresses_replay_action(self):
        policy = self.make_policy(target_clip=[0., 1.])
        batch = self.make_batch()
        policy.q_targets = ActionQ()
        candidates = torch.tensor([[[-1., 0.], [0.5, 0.], [1., 0.]],
                                   [[-1., 0.], [-0.5, 0.], [1., 0.]]])
        for discounts, expected in [(batch.discounts, torch.tensor([0., 1.])),
                                    (None, torch.tensor([-0.2 + policy.gamma * 0.25, 1.]))]:
            with self.subTest(discounts=discounts), \
                 patch.object(policy.actor, 'sample_candidates', return_value=candidates) as sample, \
                 patch('torch.randperm', return_value=torch.tensor([0, 1, 2])) as subset, \
                 patch.object(policy.qs, 'forward', wraps=policy.qs.forward) as current:
                loss, info = policy.compute_critic_loss(batch._replace(discounts=discounts))
            sample.assert_called_once()
            subset.assert_called_once()
            torch.testing.assert_close(sample.call_args.args[0],
                                       torch.cat([batch.next_observations, batch.actions[:, 4:]], -1))
            torch.testing.assert_close(current.call_args.args[1],
                                       torch.tensor([[2.1, 3.2], [3.9, 4.8]]))
            preds = policy.qs(batch.observations, current.call_args.args[1]).squeeze(-1)
            torch.testing.assert_close(loss, ((preds - expected) ** 2).mean(-1).sum())
            torch.testing.assert_close(info['q_target'], expected.mean())
            loss.backward()
            self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))
            self.assertTrue(all(not v.requires_grad for v in info.values()))
            policy.zero_grad(set_to_none=True)

    def test_switches_are_independent_and_eval_uses_candidates(self):
        batch = self.make_batch()
        actor_input = torch.cat([batch.observations, batch.actions[:, 2:4]], -1)
        for rollout in (False, True):
            for backup in (False, True):
                policy = self.make_policy(select_rollout=rollout, select_backup=backup)
                with patch.object(policy.actor, 'sample_candidates', wraps=policy.actor.sample_candidates) as candidates:
                    for argmax in (False, True):
                        torch.manual_seed(9)
                        actual = policy.predict_res_naction(actor_input, argmax)
                        if not rollout:
                            torch.manual_seed(9)
                            expected = (policy.actor.get_eval_action(actor_input) if argmax
                                        else policy.actor.get_action(actor_input)['sample'])
                            torch.testing.assert_close(actual, expected)
                    self.assertEqual(candidates.call_count, 2 if rollout else 0)
                    candidates.reset_mock()
                    policy.compute_critic_loss(batch)
                    self.assertEqual(candidates.call_count, int(backup))

    def test_actor_alpha_losses_gradients_and_rng_unchanged_by_selection(self):
        plain = self.make_policy(select_rollout=False, select_backup=False)
        selected = self.make_policy(include_base_candidate=True)
        selected.load_state_dict(plain.state_dict(), strict=True)
        for method in ('compute_actor_loss', 'compute_alpha_loss'):
            results = []
            for policy in (plain, selected):
                policy.zero_grad(set_to_none=True)
                torch.manual_seed(11)
                with patch.object(policy.actor, 'sample_candidates', side_effect=AssertionError('selection in loss')):
                    result = getattr(policy, method)(self.make_batch())
                loss = result[0] if isinstance(result, tuple) else result
                loss.backward()
                results.append((loss, torch.get_rng_state(),
                                {n: p.grad.clone() for n, p in policy.named_parameters() if p.grad is not None}))
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            self.assertTrue(torch.equal(results[0][1], results[1][1]))
            self.assertEqual(results[0][2].keys(), results[1][2].keys())
            for name in results[0][2]:
                torch.testing.assert_close(results[0][2][name], results[1][2][name], rtol=0, atol=0)

    def test_k_one_backup_matches_single_sample_loss_gradients_and_rng(self):
        plain = self.make_policy(select_rollout=False, select_backup=False)
        selected = self.make_policy(num_res_candidates=1)
        selected.load_state_dict(plain.state_dict())
        results = []
        for policy in (plain, selected):
            torch.manual_seed(13)
            loss, info = policy.compute_critic_loss(self.make_batch())
            loss.backward()
            results.append((loss, info, torch.get_rng_state(), [p.grad for p in policy.qs.parameters()]))
        torch.testing.assert_close(results[0][0], results[1][0])
        for key in results[0][1]:
            torch.testing.assert_close(results[0][1][key], results[1][1][key])
        self.assertTrue(torch.equal(results[0][2], results[1][2]))
        for left, right in zip(results[0][3], results[1][3]):
            torch.testing.assert_close(left, right)

    def test_sum_policy_interfaces_scale_once_and_mask_selected_residual(self):
        obs = torch.tensor([[1., 2.], [3., 4.]])
        base = DummyBasePolicy(obs, n_action_steps=2, action_dim=1)
        base.naction = torch.tensor([[[2.], [3.]], [[4.], [5.]]])
        res = self.make_policy(num_res_candidates=1)
        combined = SumPolicy(0.5, 2, 0, 1, 2, base, res)
        candidates = torch.tensor([[[0.2, 0.4]], [[-0.2, -0.4]]])
        with patch.object(res.actor, 'sample_candidates', return_value=candidates) as sample:
            evaluated = combined.predict_action({'state': torch.zeros(2, 1, 1)})
            trained = combined.predict_train_action(base.naction, obs, torch.tensor([False, True]))
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(set(evaluated), {'action'})
        self.assertEqual(set(trained), {'action', 'naction', 'res_naction_flat'})
        torch.testing.assert_close(evaluated['action'], base.naction + 0.5 * candidates.reshape(2, 2, 1))
        torch.testing.assert_close(trained['res_naction_flat'], torch.tensor([[0.2, 0.4], [0., 0.]]))
        torch.testing.assert_close(trained['action'][1], base.naction[1])
        torch.testing.assert_close(trained['action'][0], evaluated['action'][0])

    def test_invalid_selection_configs_and_legacy_modes(self):
        invalid = [dict(critic_action_input='base_res'), dict(actor_input='obs'), dict(q_ent=True),
                   dict(num_res_candidates=0), dict(num_res_candidates=1.5),
                   dict(num_subset=0), dict(num_subset=4)]
        for flags in [dict(select_rollout=True, select_backup=False),
                      dict(select_rollout=False, select_backup=True)]:
            for override in invalid:
                with self.subTest(flags=flags, override=override), self.assertRaises(ValueError):
                    self.make_policy(**flags, **override)
        for actor_input in ('obs', 'obs_action'):
            for critic_input in ('exec', 'base_res'):
                policy = self.make_policy(select_rollout=False, select_backup=False,
                                          actor_input=actor_input, critic_action_input=critic_input, q_ent=True)
                with patch.object(policy.actor, 'sample_candidates', side_effect=AssertionError('selector used')):
                    policy.compute_critic_loss(self.make_batch())[0].backward()
                    policy.compute_actor_loss(self.make_batch())[0].backward()
                    policy.compute_alpha_loss(self.make_batch()).backward()

    def test_config_and_eval_payload_roundtrip_no_new_state_keys(self):
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        config_dir = pathlib.Path(__file__).resolve().parents[1] / 'zprl' / 'config'
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            original = compose(config_name='train_online_robomimic_workspace')
            cfg = compose(config_name='train_online_resrl_selection_workspace', overrides=[
                'res_policy.num_base_candidates=1', 'res_policy.num_res_candidates=8',
                'res_policy.hidden_dim=8', 'res_policy.num_qs=3', 'n_action_steps=8'])
        self.assertNotIn('select_rollout', original.res_policy)
        self.assertEqual(cfg.online_task.env_runner.n_action_steps, 8)
        self.assertEqual(cfg.res_policy.gamma, round(cfg.single_gamma ** 8, 4))
        self.assertEqual(cfg.res_policy.res_scale, cfg.training.res_scale)
        policy = hydra.utils.instantiate(cfg.res_policy, obs_dim=2, action_dim=2)
        plain = self.make_policy(select_rollout=False, select_backup=False)
        self.assertEqual(policy.state_dict().keys(), plain.state_dict().keys())
        policy.load_state_dict(plain.state_dict(), strict=True)
        stream = io.BytesIO()
        torch.save({'cfg': cfg, 'res_policy': policy.state_dict()}, stream, pickle_module=dill)
        stream.seek(0)
        payload = torch.load(stream, pickle_module=dill)
        restored = hydra.utils.instantiate(payload['cfg'].res_policy, obs_dim=2, action_dim=2)
        restored.load_state_dict(payload['res_policy'], strict=True)
        self.assertTrue(restored.select_rollout and restored.select_backup and restored.include_base_candidate)
        self.assertEqual(restored.num_res_candidates, 8)
        x = torch.randn(2, 4)
        torch.manual_seed(17)
        expected = policy.predict_res_naction(x, argmax=True)
        torch.manual_seed(17)
        torch.testing.assert_close(restored.predict_res_naction(x, argmax=True), expected)


if __name__ == '__main__':
    unittest.main()
