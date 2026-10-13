import tempfile
import unittest
from pathlib import Path

import dill
import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from tests import test_reactive_history_policy as policy_tests
from zprl.model.transformer import OutputGate, OutputGatedTransformerBlock, OutputGatedTransformerEncoder
from zprl.policy.reactive_history_policy import HistoryResiduePolicy


class OutputGatedTransformerTest(unittest.TestCase):
    def make_policy(self):
        return HistoryResiduePolicy(
            obs_dim=3, action_dim=2, history_length=3, hidden_dim=8,
            d_model=8, nhead=2, num_layers=2, dim_feedforward=16,
            sub_ratio=2, critic_action_input='base_res', target_clip=[0, 1])

    def test_gate_initialization_formula_and_direct_skip(self):
        gate = OutputGate(4)
        torch.testing.assert_close(gate.weight, torch.zeros_like(gate.weight))
        torch.testing.assert_close(gate.bias, torch.full_like(gate.bias, -1))
        x = torch.randn(2, 3, 4, requires_grad=True)
        branch = torch.randn_like(x, requires_grad=True)
        torch.testing.assert_close(gate(x, branch), x + torch.sigmoid(torch.tensor(-1.)) * branch)
        torch.testing.assert_close(gate(x, torch.zeros_like(branch)), x, atol=0, rtol=0)
        with torch.no_grad():
            gate.weight.fill_(0.2)
        expected = x + torch.sigmoid(0.2 * x.sum(-1, keepdim=True) - 1) * branch
        torch.testing.assert_close(gate(x, branch), expected)
        gate(x, branch).sum().backward()
        self.assertGreater(gate.weight.grad.abs().sum().item(), 0)
        self.assertGreater(gate.bias.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(branch.grad).all())

    def test_block_preserves_pre_ln_branch_weights_and_uses_two_output_gates(self):
        torch.manual_seed(11)
        reference = torch.nn.TransformerEncoderLayer(
            d_model=8, nhead=2, dim_feedforward=16, dropout=0,
            batch_first=True, norm_first=True)
        torch.manual_seed(11)
        block = OutputGatedTransformerBlock(8, 2, 16, dropout=0)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(block.state_dict()[name], value, atol=0, rtol=0)
        x = torch.randn(2, 4, 8)
        mask = torch.ones(4, 4, dtype=torch.bool).triu(1)
        normalized = reference.norm1(x)
        attention = reference.self_attn(normalized, normalized, normalized,
                                        attn_mask=mask, need_weights=False)[0]
        scale = torch.sigmoid(torch.tensor(-1.))
        x1 = x + scale * attention
        ffn = reference.linear2(F.relu(reference.linear1(reference.norm2(x1))))
        torch.testing.assert_close(block(x, mask), x1 + scale * ffn, atol=0, rtol=0)
        self.assertNotEqual(block.attention_gate.weight.data_ptr(), block.ffn_gate.weight.data_ptr())
        with torch.no_grad():
            block.attention_gate.bias.fill_(-100)
            block.ffn_gate.bias.fill_(-100)
        x.requires_grad_()
        result = block(x, mask)
        torch.testing.assert_close(result, x, atol=0, rtol=0)
        result.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)

    def test_stacked_layers_have_independent_parameters_and_no_final_norm(self):
        encoder = OutputGatedTransformerEncoder(8, 2, 2, 16, dropout=0)
        self.assertEqual(len(encoder.layers), 2)
        self.assertFalse(hasattr(encoder, 'norm'))
        for name, parameter in encoder.layers[0].named_parameters():
            other = dict(encoder.layers[1].named_parameters())[name]
            self.assertNotEqual(parameter.data_ptr(), other.data_ptr())
            torch.testing.assert_close(parameter, other, atol=0, rtol=0)

    def test_gate_ownership_gradients_and_target_polyak_update(self):
        policy = self.make_policy()
        batch = policy_tests.ReactiveHistoryPolicyTest().make_batch()
        optimizers = policy.get_optimizer(1e-4, 1e-4)
        for module, optimizer in ((policy.actor, optimizers['actor_optimizer']),
                                   (policy.qs, optimizers['q_optimizer'])):
            owned = {id(p) for group in optimizer.param_groups for p in group['params']}
            gates = [p for name, p in module.named_parameters() if '_gate.' in name]
            self.assertEqual(len(gates), 8)
            self.assertTrue(all(id(p) in owned for p in gates))
        for method, module in (('compute_critic_loss', policy.qs),
                               ('compute_actor_loss', policy.actor)):
            policy.zero_grad(set_to_none=True)
            loss, _ = getattr(policy, method)(batch)
            loss.backward()
            for name, parameter in module.named_parameters():
                if '_gate.' in name:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            self.assertTrue(all(p.grad is None for p in policy.q_targets.parameters()))
            if method == 'compute_actor_loss':
                self.assertTrue(all(p.grad is None for p in policy.qs.parameters()))
        before = {name: p.clone() for name, p in policy.q_targets.named_parameters()
                  if '_gate.' in name}
        with torch.no_grad():
            for name, parameter in policy.qs.named_parameters():
                if '_gate.' in name:
                    parameter.add_(1)
        policy.target_update()
        online = dict(policy.qs.named_parameters())
        for name, parameter in policy.q_targets.named_parameters():
            if '_gate.' in name:
                torch.testing.assert_close(parameter,
                                           before[name] * (1 - policy.tau) + online[name] * policy.tau)
                self.assertFalse(parameter.requires_grad)
        self.assertFalse(policy.q_targets.training)

    def test_checkpoint_contains_and_restores_gates(self):
        cfg = OmegaConf.create(dict(
            _target_='zprl.policy.reactive_history_policy.HistoryResiduePolicy',
            obs_dim=3, action_dim=2, history_length=3, hidden_dim=8,
            d_model=8, nhead=2, num_layers=2, dim_feedforward=16,
            dropout=0, causal_mask=True, sub_ratio=2, num_qs=3,
            critic_action_input='base_res', target_clip=[0, 1]))
        policy = hydra.utils.instantiate(cfg).eval()
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                if '_gate.' in name:
                    parameter.add_(0.1)
        batch = policy_tests.ReactiveHistoryPolicyTest().make_batch()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'online.ckpt'
            torch.save({'cfg': cfg, 'res_policy': policy.state_dict()}, checkpoint, pickle_module=dill)
            payload = torch.load(checkpoint, pickle_module=dill)
            for prefix in ('actor.', 'qs.', 'q_targets.'):
                self.assertTrue(any(name.startswith(prefix) and '_gate.' in name
                                    for name in payload['res_policy']))
            restored = hydra.utils.instantiate(payload['cfg']).eval()
            restored.load_state_dict(payload['res_policy'])
            for module in (restored.actor, restored.qs, restored.q_targets):
                self.assertIsInstance(module.encoder.transformer, OutputGatedTransformerEncoder)
            args = (batch.obs, batch.a_exec, batch.obs_valid, batch.a_base)
            torch.testing.assert_close(restored.predict_res_naction(*args, argmax=True),
                                       policy.predict_res_naction(*args, argmax=True), atol=0, rtol=0)
            torch.testing.assert_close(restored.qs(*args, batch.action),
                                       policy.qs(*args, batch.action), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
