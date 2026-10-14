import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dill
import hydra
import torch
from omegaconf import OmegaConf

from tests import test_reactive_history_policy as policy_tests
from zprl.model.rl_history import HistoryActor, HistoryQEnsemble, HistoryMLPActor, HistoryMLPQEnsemble
from zprl.policy.reactive_history_policy import HistoryResiduePolicy


class ReactiveHistoryStageATest(unittest.TestCase):
    def make_policy(self, gru=False):
        policy = HistoryResiduePolicy(
            obs_dim=3, action_dim=2, history_length=3, hidden_dim=8,
            num_qs=3, res_scale=0.05, critic_action_input='base_res',
            gru_input_dim=8, gru_hidden_dim=8, gru_num_layers=2, sub_ratio=1)
        if not gru:
            common = dict(obs_dim=3, exec_dim=2, history_length=3, hidden_dim=8)
            policy.actor = HistoryMLPActor(action_dim=2, **common)
            policy.qs = HistoryMLPQEnsemble(num_qs=3, critic_action_input='base_res', **common)
            policy.q_targets = copy.deepcopy(policy.qs).requires_grad_(False).eval()
        return policy

    def test_mlp_and_gru_head_receive_base_and_unscaled_residual(self):
        for gru in (False, True):
            for length in (1, 2, 4):
                with self.subTest(gru=gru, history_length=length):
                    common = dict(obs_dim=3, exec_dim=2, num_qs=3, history_length=length,
                                  hidden_dim=8, critic_action_input='base_res')
                    if gru:
                        qs = HistoryQEnsemble(**common, gru_input_dim=8, gru_hidden_dim=8,
                                             gru_num_layers=2, sub_ratio=1).eval()
                    else:
                        qs = HistoryMLPQEnsemble(**common).eval()
                    obs = torch.randn(2, length, 3)
                    history_exec = torch.randn(2, length, 2)
                    valid = torch.ones(2, length, dtype=torch.bool)
                    valid[0, :-1] = False
                    obs[~valid] = float('nan')
                    history_exec[~valid] = float('nan')
                    history_exec[:, -1] = float('nan')
                    base = torch.randn(2, 2)
                    residual = torch.rand(2, 2, requires_grad=True)
                    captured = []
                    hook = qs.net[0].register_forward_pre_hook(
                        lambda module, args: captured.append(args[0].detach().clone()))
                    values = qs(obs, history_exec, valid, base, residual)
                    hook.remove()
                    history = qs.encoder(obs, history_exec, valid)
                    if gru:
                        history = torch.cat([history, obs[:, -1]], dim=-1)
                    expected = torch.cat([history, base, residual], dim=-1)
                    torch.testing.assert_close(captured[0], expected[None].expand(3, -1, -1))
                    self.assertEqual(qs.net[0].weight.shape[1], history.shape[-1] + 4)
                    self.assertEqual(values.shape, (3, 2, 1))
                    values.sum().backward()
                    self.assertTrue(torch.isfinite(residual.grad).all())
                    self.assertGreater(residual.grad.abs().sum().item(), 0)
                    with self.assertRaises(AssertionError):
                        qs(obs, history_exec, valid, base)

    def test_regression_and_n_step_bootstrap_use_separate_endpoint_base(self):
        policy = self.make_policy()
        policy.num_subset = policy.num_qs
        policy.q_ent = False
        batch = policy_tests.ReactiveHistoryPolicyTest().make_batch()
        next_res = torch.full_like(batch.action, 0.3)
        target_values = torch.tensor([[[3.], [5.]], [[4.], [6.]], [[7.], [8.]]])
        current_inputs, target_inputs = [], []
        original_q = policy.qs.forward

        def current_forward(obs, a_exec, valid, base, residual):
            current_inputs.append((obs, a_exec, valid, base, residual))
            return original_q(obs, a_exec, valid, base, residual)

        def target_forward(obs, a_exec, valid, base, residual):
            target_inputs.append((obs, a_exec, valid, base, residual))
            self.assertFalse(torch.is_grad_enabled())
            return target_values

        with patch.object(policy, '_sample_naction_log_prob',
                          return_value=(next_res, torch.zeros(2, 1), None)), \
             patch.object(policy.qs, 'forward', side_effect=current_forward), \
             patch.object(policy.q_targets, 'forward', side_effect=target_forward):
            loss, info = policy.compute_critic_loss(batch)
        for actual, expected in zip(current_inputs[0],
                                     (batch.obs, batch.a_exec, batch.obs_valid,
                                      batch.a_base, batch.action)):
            self.assertIs(actual, expected)
        for actual, expected in zip(target_inputs[0],
                                     (batch.next_obs, batch.next_a_exec, batch.next_obs_valid,
                                      batch.next_a_base, next_res)):
            self.assertIs(actual, expected)
        self.assertAlmostEqual(info['q_target'].item(), 2.7, places=5)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in policy.q_targets.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))
        policy.q_ent = True
        policy.auto_alpha = False
        policy.init_alpha = 0.25
        with patch.object(policy, '_sample_naction_log_prob',
                          return_value=(next_res, torch.full((2, 1), -2.), None)), \
             patch.object(policy.q_targets, 'forward', side_effect=target_forward):
            _, info = policy.compute_critic_loss(batch)
        self.assertAlmostEqual(info['q_target'].item(), 2.9, places=5)

    def test_actor_gradient_reaches_raw_residual_with_frozen_q(self):
        for gru in (False, True):
            with self.subTest(gru=gru):
                policy = self.make_policy(gru)
                policy.auto_alpha = False
                policy.init_alpha = 0
                batch = policy_tests.ReactiveHistoryPolicyTest().make_batch()
                captured = []
                original_q = policy.qs.forward

                def current_forward(obs, a_exec, valid, base, residual):
                    self.assertIs(base, batch.a_base)
                    self.assertTrue(residual.requires_grad)
                    self.assertTrue(torch.all(residual.abs() <= 1))
                    self.assertTrue(all(not p.requires_grad for p in policy.qs.parameters()))
                    residual.retain_grad()
                    captured.append(residual)
                    return original_q(obs, a_exec, valid, base, residual)

                with patch.object(policy.qs, 'forward', side_effect=current_forward):
                    loss, _ = policy.compute_actor_loss(batch)
                loss.backward()
                self.assertGreater(captured[0].grad.abs().sum().item(), 0)
                self.assertGreater(policy.actor.net[0].weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None for p in policy.qs.parameters()))
                self.assertTrue(all(p.requires_grad for p in policy.qs.parameters()))

    def test_checkpoint_mode_roundtrip_and_legacy_shape_compatibility(self):
        for mode in ('exec', 'base_res'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                cfg = OmegaConf.create(dict(
                    _target_='zprl.policy.reactive_history_policy.HistoryResiduePolicy',
                    obs_dim=3, action_dim=2, history_length=3, hidden_dim=8, num_qs=3))
                if mode == 'base_res':
                    cfg.critic_action_input = mode
                policy = hydra.utils.instantiate(cfg).eval()
                self.assertEqual(policy.critic_action_input, mode)
                path = Path(tmp) / 'online.ckpt'
                torch.save({'cfg': cfg, 'res_policy': policy.state_dict()}, path, pickle_module=dill)
                payload = torch.load(path, pickle_module=dill)
                restored = hydra.utils.instantiate(payload['cfg']).eval()
                restored.load_state_dict(payload['res_policy'])
                batch = policy_tests.ReactiveHistoryPolicyTest().make_batch()
                torch.manual_seed(13)
                loss, _ = policy.compute_critic_loss(batch)
                torch.manual_seed(13)
                restored_loss, _ = restored.compute_critic_loss(batch)
                torch.testing.assert_close(loss, restored_loss, atol=0, rtol=0)
                incompatible = self.make_policy() if mode == 'exec' else \
                    hydra.utils.instantiate(OmegaConf.merge(cfg, {'critic_action_input': 'exec'}))
                with self.assertRaises(RuntimeError):
                    incompatible.load_state_dict(payload['res_policy'])

    def test_invalid_mode_is_rejected(self):
        for cls in (HistoryQEnsemble, HistoryMLPQEnsemble):
            with self.assertRaises(ValueError):
                cls(3, 2, 3, 3, critic_action_input='invalid')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_losses_and_backward_for_both_encoders(self):
        for gru in (False, True):
            with self.subTest(gru=gru):
                policy = self.make_policy(gru).cuda()
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
