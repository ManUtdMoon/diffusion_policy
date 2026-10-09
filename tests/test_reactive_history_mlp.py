import unittest

import torch
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from tests import test_reactive_history_policy as policy_tests
from zprl.model.online import Actor, BatchedSoftQNet
from zprl.model.rl_history import FlattenHistoryEncoder, HistoryMLPActor, HistoryMLPQEnsemble
from zprl.policy.reactive_history_policy import HistoryResiduePolicy
from zprl.policy.residue_policy import ResiduePolicy


class ReactiveHistoryMLPTest(unittest.TestCase):
    def test_flatten_order_padding_and_no_current_action_slot(self):
        encoder = FlattenHistoryEncoder(2, 1, 3, debug=True)
        obs = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])
        actions = torch.tensor([[[10.], [20.], [float('nan')]]])
        valid = torch.ones(1, 3, dtype=torch.bool)
        torch.testing.assert_close(encoder(obs, actions, valid),
                                   torch.tensor([[1., 2., 3., 4., 5., 6.]]))
        self.assertEqual(encoder.output_dim, 6)
        valid[:, 0] = False
        obs[:, 0] = float('nan')
        actions[:, 0] = float('nan')
        torch.testing.assert_close(encoder(obs, actions, valid),
                                   torch.tensor([[0., 0., 3., 4., 5., 6.]]))
        current = FlattenHistoryEncoder(2, 1, 1)
        torch.testing.assert_close(current(obs[:, -1:], actions[:, -1:], valid[:, -1:]),
                                   torch.tensor([[5., 6.]]))
        valid[:, -1] = False
        with self.assertRaises(AssertionError):
            encoder(obs, actions, valid)

    def test_original_mlp_weights_outputs_and_history_gradients(self):
        for length in (1, 2, 4):
            with self.subTest(history_length=length):
                obs = torch.randn(2, length, 3, requires_grad=True)
                actions = torch.randn(2, length, 2, requires_grad=True)
                valid = torch.ones(2, length, dtype=torch.bool)
                valid[0, :-1] = False
                base = torch.randn(2, 2)
                candidate = torch.randn(2, 2, requires_grad=True)
                actor = HistoryMLPActor(3, 2, 2, length, hidden_dim=8)
                qs = HistoryMLPQEnsemble(3, 2, 3, length, hidden_dim=8)
                reference_actor = Actor(actor.encoder.output_dim + 2, 2, hidden_dim=8)
                reference_q = BatchedSoftQNet(qs.encoder.output_dim, 2, 3, hidden_dim=8)
                actor.load_state_dict(reference_actor.state_dict())
                qs.load_state_dict(reference_q.state_dict())
                flat = actor.encoder(obs, actions, valid)
                expected = reference_actor(torch.cat([flat, base], dim=-1))
                for actual, reference in zip(actor(obs, actions, valid, base), expected):
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                torch.manual_seed(42)
                result = actor.get_action(obs, actions, valid, base)
                torch.manual_seed(42)
                reference = reference_actor.get_action(torch.cat([flat, base], dim=-1))
                for key in result:
                    torch.testing.assert_close(result[key], reference[key], atol=0, rtol=0)
                value = qs(obs, actions, valid, candidate)
                torch.testing.assert_close(value, reference_q(flat, candidate), atol=0, rtol=0)
                changed_actions = torch.full_like(actions, float('nan'))
                for actual, changed in zip(actor(obs, actions, valid, base),
                                           actor(obs, changed_actions, valid, base)):
                    torch.testing.assert_close(actual, changed, atol=0, rtol=0)
                torch.testing.assert_close(value, qs(obs, changed_actions, valid, candidate),
                                           atol=0, rtol=0)
                (value.sum() + result['sample'].sum()).backward()
                self.assertGreater(candidate.grad.abs().sum().item(), 0)
                self.assertIsNone(actions.grad)
                torch.testing.assert_close(obs.grad[~valid], torch.zeros_like(obs.grad[~valid]))
                if length > 1:
                    self.assertGreater(obs.grad[1, :-1].abs().sum().item(), 0)

    def test_sac_loss_gradients_and_optimizer_step_match_original(self):
        for length in (1, 3):
            with self.subTest(history_length=length):
                batch = policy_tests.ReactiveHistoryPolicyTest().make_batch()
                if length == 1:
                    batch = batch._replace(
                        obs=batch.obs[:, -1:], a_exec=batch.a_exec[:, -1:],
                        obs_valid=batch.obs_valid[:, -1:],
                        next_obs=batch.next_obs[:, -1:], next_a_exec=batch.next_a_exec[:, -1:],
                        next_obs_valid=batch.next_obs_valid[:, -1:])
                policy = HistoryResiduePolicy(3, 2, length, hidden_dim=8,
                                              num_qs=3, res_scale=0.5)
                reference = ResiduePolicy(policy.actor.encoder.output_dim, 2,
                                          hidden_dim=8, num_qs=3, res_scale=0.5)
                policy.load_state_dict(reference.state_dict())
                legacy = ReplayBufferSamples(
                    observations=policy.actor.encoder(batch.obs, batch.a_exec, batch.obs_valid),
                    actions=torch.cat([batch.action, batch.a_base, batch.next_a_base], dim=-1),
                    next_observations=policy.actor.encoder(
                        batch.next_obs, batch.next_a_exec, batch.next_obs_valid),
                    dones=batch.dones, rewards=batch.rewards, discounts=batch.discounts)
                optimizers = policy.get_optimizer(1e-3, 1e-3)
                reference_optimizers = reference.get_optimizer(1e-3, 1e-3)
                for method, name in [('compute_critic_loss', 'q_optimizer'),
                                     ('compute_actor_loss', 'actor_optimizer'),
                                     ('compute_alpha_loss', 'alpha_optimizer')]:
                    policy.zero_grad(set_to_none=True)
                    reference.zero_grad(set_to_none=True)
                    torch.manual_seed(17)
                    actual = getattr(policy, method)(batch)
                    torch.manual_seed(17)
                    expected = getattr(reference, method)(legacy)
                    if isinstance(actual, tuple):
                        actual_loss, actual_info = actual
                        expected_loss, expected_info = expected
                        for key in actual_info:
                            torch.testing.assert_close(actual_info[key],
                                                       torch.as_tensor(expected_info[key]))
                    else:
                        actual_loss, expected_loss = actual, expected
                    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
                    actual_loss.backward()
                    expected_loss.backward()
                    actual_params = optimizers[name].param_groups[0]['params']
                    expected_params = reference_optimizers[name].param_groups[0]['params']
                    for actual_param, expected_param in zip(actual_params, expected_params):
                        torch.testing.assert_close(actual_param.grad, expected_param.grad,
                                                   atol=0, rtol=0)
                    optimizers[name].step()
                    reference_optimizers[name].step()
                    for actual_param, expected_param in zip(actual_params, expected_params):
                        torch.testing.assert_close(actual_param, expected_param, atol=0, rtol=0)
                policy.target_update()
                reference.target_update()
                for actual_param, expected_param in zip(policy.q_targets.parameters(),
                                                        reference.q_targets.parameters()):
                    torch.testing.assert_close(actual_param, expected_param)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_history_loss_and_backward(self):
        policy = HistoryResiduePolicy(3, 2, 3, hidden_dim=8, num_qs=3).cuda()
        batch = policy_tests.ReactiveHistoryPolicyTest().make_batch()
        batch = batch._replace(**{name: value.cuda() for name, value in batch._asdict().items()
                                  if isinstance(value, torch.Tensor)})
        for method in ('compute_critic_loss', 'compute_actor_loss', 'compute_alpha_loss'):
            policy.zero_grad(set_to_none=True)
            result = getattr(policy, method)(batch)
            loss = result[0] if isinstance(result, tuple) else result
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters()
                                if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
