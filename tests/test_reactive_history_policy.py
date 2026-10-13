import unittest
from unittest.mock import patch

import torch

from zprl.common.reactive_history import HistoryReplayBufferSamples
from zprl.policy.reactive_history_policy import HistoryResiduePolicy, HistorySumPolicy


class IdentityNormalizer:
    def unnormalize(self, action):
        return action


class DummyBasePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.normalizer = {'action': IdentityNormalizer()}

    def encode_obs(self, obs):
        return obs['state'][:, -1]

    def predict_action(self, obs):
        feature = self.encode_obs(obs)
        plan = torch.arange(4, dtype=feature.dtype, device=feature.device)
        return {'obs_emb': feature, 'naction': plan[None, :, None].expand(feature.shape[0], -1, -1)}


class ReactiveHistoryPolicyTest(unittest.TestCase):
    def make_policy(self, tau=0.2, target_clip=None):
        return HistoryResiduePolicy(
            obs_dim=3, action_dim=2, history_length=3, hidden_dim=8,
            num_qs=3, num_subset=2, tau=tau,
            res_scale=0.5, target_clip=target_clip, d_model=8, nhead=2,
            num_layers=2, dim_feedforward=16)

    def make_batch(self):
        obs = torch.randn(2, 3, 3)
        next_obs = torch.randn(2, 3, 3)
        valid = torch.tensor([[False, True, True], [True, True, True]])
        return HistoryReplayBufferSamples(
            obs=obs, a_base=torch.randn(2, 2), a_exec=torch.randn(2, 3, 2),
            obs_valid=valid, exec_steps=torch.tensor([[0, 2, 0], [2, 2, 0]]),
            next_obs=next_obs, next_a_base=torch.randn(2, 2),
            next_a_exec=torch.randn(2, 3, 2), next_obs_valid=valid.clone(),
            next_exec_steps=torch.tensor([[0, 2, 0], [2, 2, 0]]),
            action=torch.randn(2, 2), rewards=torch.tensor([[1.], [2.]]),
            dones=torch.tensor([[0.], [1.]]), discounts=torch.tensor([[0.8], [0.6]]))

    def test_optimizer_ownership_and_real_history_step(self):
        policy = self.make_policy()
        batch = self.make_batch()
        optimizers = policy.get_optimizer(1e-2, 1e-2)
        for key, module in (('actor_optimizer', policy.actor), ('q_optimizer', policy.qs)):
            optimizer_ids = {id(p) for group in optimizers[key].param_groups for p in group['params']}
            self.assertEqual(optimizer_ids, {id(p) for p in module.parameters()})
        self.assertEqual([p for group in optimizers['alpha_optimizer'].param_groups
                          for p in group['params']], [policy.log_alpha])
        self.assertTrue(all(not p.requires_grad for p in policy.q_targets.parameters()))
        self.assertFalse(policy.q_targets.training)

        before = policy.qs.net[0].weight.detach().clone()
        loss, info = policy.compute_critic_loss(batch)
        optimizers['q_optimizer'].zero_grad()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('q_target', info)
        self.assertGreater(policy.qs.net[0].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(policy.actor.net[0].weight.grad)
        self.assertTrue(all(p.grad is None for p in policy.q_targets.parameters()))
        optimizers['q_optimizer'].step()
        self.assertFalse(torch.equal(before, policy.qs.net[0].weight))

    def test_deferred_metrics_and_debug_preserve_losses_and_gradients(self):
        policy = self.make_policy()
        checked = self.make_policy()
        checked.load_state_dict(policy.state_dict())
        for module in (checked.actor.encoder, checked.qs.encoder, checked.q_targets.encoder):
            module.debug = True
        batch = self.make_batch()
        for method in ('compute_critic_loss', 'compute_actor_loss'):
            policy.zero_grad()
            checked.zero_grad()
            torch.manual_seed(42)
            with patch.object(torch.Tensor, 'item', side_effect=AssertionError('eager item')):
                loss, info = getattr(policy, method)(batch)
            torch.manual_seed(42)
            checked_loss, checked_info = getattr(checked, method)(batch)
            torch.testing.assert_close(loss, checked_loss, atol=0, rtol=0)
            loss.backward()
            checked_loss.backward()
            self.assertIsNone(policy.log_alpha.grad)
            for parameter, reference in zip(policy.parameters(), checked.parameters()):
                if reference.grad is None:
                    self.assertIsNone(parameter.grad)
                else:
                    torch.testing.assert_close(parameter.grad, reference.grad, atol=0, rtol=0)
            for key, value in info.items():
                self.assertIsInstance(value, torch.Tensor)
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
                torch.testing.assert_close(value, checked_info[key], atol=0, rtol=0)

    def test_actor_action_gradient_without_critic_parameter_gradient(self):
        policy = self.make_policy()
        policy.auto_alpha = False
        policy.init_alpha = 0
        batch = self.make_batch()
        policy.actor.eval()
        policy.qs.train()
        before = policy.actor.net[0].weight.detach().clone()
        actor_loss, _ = policy.compute_actor_loss(batch)
        self.assertFalse(policy.actor.training)
        self.assertTrue(policy.qs.training)
        actor_loss.backward()
        self.assertGreater(policy.actor.net[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in policy.qs.parameters()))
        self.assertTrue(all(p.requires_grad for p in policy.qs.parameters()))
        policy.get_optimizer(1e-2, 1e-2)['actor_optimizer'].step()
        self.assertFalse(torch.equal(before, policy.actor.net[0].weight))
        candidate = torch.randn(2, 2, requires_grad=True)
        q_value = policy.qs(batch.obs, batch.a_exec, batch.obs_valid, candidate)
        q_value.sum().backward()
        self.assertGreater(candidate.grad.abs().sum().item(), 0)
        changed = policy.qs(batch.obs, batch.a_exec, batch.obs_valid, candidate.detach() + 1)
        self.assertGreater((changed - q_value).abs().max().item(), 1e-6)

    def test_modes_restore_and_target_stays_eval(self):
        policy = self.make_policy()
        batch = self.make_batch()
        policy.train()
        self.assertFalse(policy.q_targets.training)
        actor_modes = []
        target_modes = []
        critic_modes = []
        original_actor = policy.actor.get_action
        original_target = policy.q_targets.forward
        original_q = policy.qs.forward

        def actor_forward(*args):
            actor_modes.append(policy.actor.training)
            return original_actor(*args)

        def target_forward(*args):
            target_modes.append(policy.q_targets.training)
            return original_target(*args)

        def q_forward(*args):
            critic_modes.append(policy.qs.training)
            return original_q(*args)

        with patch.object(policy.actor, 'get_action', side_effect=actor_forward), \
             patch.object(policy.q_targets, 'forward', side_effect=target_forward), \
             patch.object(policy.qs, 'forward', side_effect=q_forward):
            policy.compute_critic_loss(batch)
            policy.compute_actor_loss(batch)
            policy.compute_alpha_loss(batch)
        self.assertEqual(actor_modes, [False, True, True])
        self.assertEqual(target_modes, [False])
        self.assertEqual(critic_modes, [True, False])
        self.assertTrue(policy.actor.training)
        self.assertTrue(policy.qs.training)
        self.assertFalse(policy.q_targets.training)
        policy.eval()
        policy.predict_res_naction(batch.obs, batch.a_exec, batch.obs_valid,
                                   batch.a_base, argmax=False)
        self.assertFalse(policy.actor.training)

    def test_polyak_updates_entire_target(self):
        policy = self.make_policy(tau=0.2)
        for parameter in policy.qs.parameters():
            parameter.data.fill_(1)
        for parameter in policy.q_targets.parameters():
            parameter.data.zero_()
        policy.target_update()
        for parameter in policy.q_targets.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.2))
            self.assertFalse(parameter.requires_grad)
        self.assertFalse(policy.q_targets.training)

    def test_metadata_isolation_and_named_sample_fields(self):
        policy = self.make_policy()
        policy.eval()
        batch = self.make_batch()
        changed = batch._replace(exec_steps=torch.full_like(batch.exec_steps, 99),
                                 next_exec_steps=torch.full_like(batch.next_exec_steps, 99))
        torch.manual_seed(4)
        loss, _ = policy.compute_critic_loss(batch)
        torch.manual_seed(4)
        changed_loss, _ = policy.compute_critic_loss(changed)
        torch.testing.assert_close(loss, changed_loss)
        q_original = policy.qs(batch.obs, batch.a_exec, batch.obs_valid,
                               batch.a_base + policy.res_scale * batch.action)
        q_changed = policy.qs(changed.obs, changed.a_exec, changed.obs_valid,
                              changed.a_base + policy.res_scale * changed.action)
        torch.testing.assert_close(q_original, q_changed)
        self.assertEqual(batch.action.shape[-1], policy.action_dim)
        self.assertEqual(batch.obs_valid.dtype, torch.bool)

    def test_n_step_target_and_current_exec_composition(self):
        policy = self.make_policy()
        policy.num_subset = policy.num_qs
        policy.q_ent = False
        batch = self.make_batch()
        next_res = torch.zeros(2, 2)
        next_log_prob = torch.zeros(2, 1)
        target_values = torch.tensor([[[3.], [5.]], [[4.], [6.]], [[7.], [8.]]])
        current_exec = []
        target_exec = []
        original_q = policy.qs.forward

        def current_forward(obs, a_exec, obs_valid, candidate):
            current_exec.append(candidate.detach().clone())
            return original_q(obs, a_exec, obs_valid, candidate)

        def target_forward(obs, a_exec, obs_valid, candidate):
            target_exec.append(candidate.detach().clone())
            return target_values

        with patch.object(policy, '_sample_naction_log_prob',
                          return_value=(next_res, next_log_prob, None)), \
             patch.object(policy.q_targets, 'forward', side_effect=target_forward), \
             patch.object(policy.qs, 'forward', side_effect=current_forward):
            _, info = policy.compute_critic_loss(batch)
        torch.testing.assert_close(current_exec[0], batch.a_base + 0.5 * batch.action)
        torch.testing.assert_close(target_exec[0], batch.next_a_base)
        self.assertAlmostEqual(info['q_target'].item(), 2.7, places=5)

        policy.q_ent = True
        policy.auto_alpha = False
        policy.init_alpha = 0.25
        with patch.object(policy, '_sample_naction_log_prob',
                          return_value=(next_res, torch.full((2, 1), -2.), None)), \
             patch.object(policy.q_targets, 'forward', return_value=target_values):
            _, entropy_info = policy.compute_critic_loss(batch)
        self.assertAlmostEqual(entropy_info['q_target'].item(), 2.9, places=5)
        policy.q_ent = False

        same_exec = batch._replace(a_base=batch.a_base + 1,
                                   action=batch.action - 2)
        policy.eval()
        torch.manual_seed(7)
        original_loss, _ = policy.compute_critic_loss(batch)
        torch.manual_seed(7)
        same_exec_loss, _ = policy.compute_critic_loss(same_exec)
        torch.testing.assert_close(original_loss, same_exec_loss)

    def test_target_clip_applies_to_complete_critic_targets(self):
        batch = self.make_batch()
        batch = batch._replace(**{name: torch.cat([value, value], dim=0)
                                  for name, value in batch._asdict().items()})
        batch = batch._replace(
            rewards=torch.tensor([[-4.], [0.], [1.], [4.]]),
            dones=torch.tensor([[0.], [0.], [0.], [1.]]),
            discounts=torch.tensor([[0.8], [0.4], [0.9], [0.6]]))
        target_values = torch.tensor([[[2.], [1.], [3.], [100.]]]).expand(3, -1, -1)
        for target_clip, expected in ((None, [-2., 0.6, 4.15, 4.]),
                                      ([-1., 2.], [-1., 0.6, 2., 2.])):
            with self.subTest(target_clip=target_clip):
                policy = self.make_policy(target_clip=target_clip)
                policy.auto_alpha = False
                policy.init_alpha = 0.25
                original_q = policy.qs.forward
                with patch.object(policy, '_sample_naction_log_prob',
                                  return_value=(torch.zeros(4, 2), torch.full((4, 1), -2.), None)), \
                     patch.object(policy.q_targets, 'forward', return_value=target_values), \
                     patch.object(policy.qs, 'forward', side_effect=lambda *args: original_q(*args) + 10), \
                     patch('zprl.policy.reactive_history_policy.F.mse_loss',
                           wraps=torch.nn.functional.mse_loss) as mse_loss:
                    loss, info = policy.compute_critic_loss(batch)
                predictions, targets = mse_loss.call_args.args[:2]
                expected = torch.tensor(expected)[None].expand(policy.num_qs, -1)
                torch.testing.assert_close(targets, expected)
                torch.testing.assert_close(info['q_target'], expected.mean())
                self.assertGreater(predictions.min().item(), 2)
                self.assertFalse(targets.requires_grad)
                loss.backward()
                self.assertGreater(policy.qs.net[0].weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))
                self.assertTrue(all(p.grad is None for p in policy.q_targets.parameters()))
                self.assertIsNone(policy.log_alpha.grad)

    def test_target_clip_does_not_change_actor_or_alpha_loss(self):
        policy = self.make_policy()
        clipped = self.make_policy(target_clip=[0., 0.])
        clipped.load_state_dict(policy.state_dict())
        batch = self.make_batch()
        for method in ('compute_actor_loss', 'compute_alpha_loss'):
            torch.manual_seed(42)
            expected = getattr(policy, method)(batch)
            torch.manual_seed(42)
            actual = getattr(clipped, method)(batch)
            expected = expected[0] if isinstance(expected, tuple) else expected
            actual = actual[0] if isinstance(actual, tuple) else actual
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_invalid_target_clip_is_rejected(self):
        for bounds in ([0.], [0., 1., 2.], [2., -1.]):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                self.make_policy(target_clip=bounds)

    def test_alpha_optimizer_and_sum_policy_freeze(self):
        policy = self.make_policy()
        batch = self.make_batch()
        alpha_loss = policy.compute_alpha_loss(batch)
        alpha_loss.backward()
        self.assertIsNotNone(policy.log_alpha.grad)
        self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))

        base = DummyBasePolicy()
        combined = HistorySumPolicy(0.5, 3, 0, 1, 4, base, policy, 2)
        combined.train()
        self.assertFalse(base.training)
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))
        state = torch.randn(2, 1, 3)
        combined.reset({'state': state})
        result = combined.predict_action(
            {'state': state}, deterministic=False, res_mask=torch.tensor([True, False]))
        base_action = torch.tensor([[[0.], [1.]], [[0.], [1.]]])
        self.assertEqual(result['action'].shape, (2, 2, 1))
        torch.testing.assert_close(result['naction'][0], base_action[0])
        torch.testing.assert_close(result['res_naction_flat'][0], torch.zeros(2))
        torch.testing.assert_close(result['naction'][1].flatten(),
                                   base_action[1].flatten() +
                                   0.5 * result['res_naction_flat'][1])

    def test_sum_policy_uses_cached_base_segments(self):
        policy = self.make_policy()
        combined = HistorySumPolicy(0.0, 3, 0, 1, 4,
                                    DummyBasePolicy(), policy, 2)
        state = torch.randn(1, 1, 3)
        combined.reset({'state': state})
        first = combined.predict_action({'state': state})
        combined.on_env_step({'state': state}, [False],
                             [{'executed_action_steps': 2}])
        second = combined.predict_action({'state': state})
        torch.testing.assert_close(first['action'].flatten(), torch.tensor([0., 1.]))
        torch.testing.assert_close(second['action'].flatten(), torch.tensor([2., 3.]))
        combined.reset({'state': state})
        reset = combined.predict_action({'state': state})
        torch.testing.assert_close(reset['action'].flatten(), first['action'].flatten())


if __name__ == '__main__':
    unittest.main()
