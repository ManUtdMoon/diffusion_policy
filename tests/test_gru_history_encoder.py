import copy
import math
import unittest

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from zprl.model.gru_history_encoder import GRUHistoryEncoder
from tests import test_reactive_history_policy as policy_tests


class GRUHistoryEncoderTest(unittest.TestCase):
    def make_encoder(self, length=7, ratio=2):
        return GRUHistoryEncoder(3, 2, length, gru_input_dim=5, gru_hidden_dim=4,
                                 gru_num_layers=2, sub_ratio=ratio, debug=True)

    def test_shapes_subsample_indices_and_current_action_slot(self):
        for length in (1, 2, 3, 4, 7):
            for ratio in (1, 2):
                with self.subTest(length=length, ratio=ratio):
                    encoder = GRUHistoryEncoder(3, 2, length, sub_ratio=ratio)
                    obs = torch.arange(2 * length * 3, dtype=torch.float32).reshape(2, length, 3)
                    actions = torch.randn(2, length, 2)
                    actions[:, -1] = float('nan')
                    valid = torch.ones(2, length, dtype=torch.bool)
                    captured, counts = [], [0, 0]
                    hook = encoder.input_projection.register_forward_pre_hook(
                        lambda module, args: captured.append(args[0]))
                    hooks = []
                    for layer, cell in enumerate(encoder.cells):
                        def count(module, args, output, layer=layer):
                            counts[layer] += 1
                        hooks.append(cell.register_forward_hook(count))
                    output = encoder(obs, actions, valid)
                    hook.remove()
                    for hook in hooks:
                        hook.remove()
                    indices = [index for index in range(length) if (length - 1 - index) % ratio == 0]
                    expected_actions = actions[:, indices].clone()
                    expected_actions[:, -1] = 0
                    torch.testing.assert_close(captured[0], torch.cat([obs[:, indices], expected_actions], -1))
                    self.assertEqual(output.shape, (2, 128))
                    self.assertEqual(encoder.output_dim, 128)
                    self.assertEqual(counts, [len(indices), len(indices)])
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue((output.abs() <= 1).all())

    def compare_packed(self, ratio, dtype, device, rtol, atol):
        torch.manual_seed(17)
        encoder = self.make_encoder(ratio=ratio).to(device=device, dtype=dtype)
        projection = copy.deepcopy(encoder.input_projection)
        output_projection = copy.deepcopy(encoder.output_projection)
        reference = torch.nn.GRU(5, 4, num_layers=2, dropout=0, batch_first=True).to(device=device, dtype=dtype)
        with torch.no_grad():
            for layer, cell in enumerate(encoder.cells):
                for name in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
                    getattr(reference, f'{name}_l{layer}').copy_(getattr(cell, name))
        obs = (torch.randn(4, 7, 3, device=device, dtype=dtype) * 0.2).requires_grad_()
        actions = (torch.randn(4, 7, 2, device=device, dtype=dtype) * 0.2).requires_grad_()
        reference_obs = obs.detach().clone().requires_grad_()
        reference_actions = actions.detach().clone().requires_grad_()
        valid = torch.tensor([[0, 0, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1, 1],
                              [0, 0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]],
                             device=device, dtype=torch.bool)
        final_states = [[], []]
        hooks = []
        for layer, cell in enumerate(encoder.cells):
            hooks.append(cell.register_forward_hook(
                lambda module, args, output, layer=layer: final_states[layer].append(output)))
        actual = encoder(obs, actions, valid)
        for hook in hooks:
            hook.remove()
        indices = [index for index in range(7) if (6 - index) % ratio == 0]
        cleaned_actions = torch.cat([reference_actions[:, :-1],
                                     torch.zeros_like(reference_actions[:, -1:])], 1)
        sequences = []
        for batch in range(4):
            kept = [index for index in indices if valid[batch, index]]
            tokens = torch.cat([reference_obs[batch, kept], cleaned_actions[batch, kept]], -1)
            sequences.append(projection(tokens))
        packed = pack_padded_sequence(pad_sequence(sequences, batch_first=True),
                                      torch.tensor([len(sequence) for sequence in sequences]),
                                      batch_first=True, enforce_sorted=False)
        _, expected_states = reference(packed)
        expected = torch.tanh(output_projection(expected_states[-1]))
        for layer in range(2):
            torch.testing.assert_close(final_states[layer][-1], expected_states[layer], rtol=rtol, atol=atol)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        weights = torch.randn_like(actual)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        for left, right in ((obs.grad, reference_obs.grad), (actions.grad, reference_actions.grad)):
            torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
        for layer, cell in enumerate(encoder.cells):
            for name in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
                torch.testing.assert_close(getattr(cell, name).grad,
                                           getattr(reference, f'{name}_l{layer}').grad, rtol=rtol, atol=atol)
        for left, right in ((encoder.input_projection, projection),
                            (encoder.output_projection, output_projection)):
            for name in ('weight', 'bias'):
                torch.testing.assert_close(getattr(left, name).grad, getattr(right, name).grad,
                                           rtol=rtol, atol=atol)

    def test_packed_gru_hidden_embedding_and_gradients_cpu_float64(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                self.compare_packed(ratio, torch.float64, 'cpu', 1e-8, 1e-10)

    def test_masked_nan_inputs_padding_and_skipped_updates(self):
        torch.manual_seed(4)
        encoder = self.make_encoder(ratio=1)
        obs, actions = torch.randn(2, 7, 3), torch.randn(2, 7, 2)
        valid = torch.tensor([[0, 0, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1, 1]], dtype=torch.bool)
        expected = encoder(obs, actions, valid)
        changed_obs, changed_actions = obs.clone(), actions.clone()
        changed_obs[~valid] = float('nan')
        changed_actions[~valid] = float('nan')
        changed_actions[:, -1] = float('nan')
        torch.testing.assert_close(encoder(changed_obs, changed_actions, valid), expected, atol=0, rtol=0)
        encoder.zero_grad(set_to_none=True)
        expected.sum().backward()
        expected_grads = {name: p.grad.clone() for name, p in encoder.named_parameters()}
        encoder.zero_grad(set_to_none=True)
        encoder(changed_obs, changed_actions, valid).sum().backward()
        for name, parameter in encoder.named_parameters():
            torch.testing.assert_close(parameter.grad, expected_grads[name], atol=0, rtol=0)
        for ratio, length, indices in ((1, 3, [4, 5, 6]), (2, 3, [4, 6])):
            long = self.make_encoder(ratio=ratio)
            short = self.make_encoder(length=length, ratio=ratio)
            short.load_state_dict(long.state_dict())
            padded_valid = torch.zeros_like(valid)
            padded_valid[:, indices] = True
            torch.testing.assert_close(long(obs, actions, padded_valid),
                                       short(obs[:, -length:], actions[:, -length:],
                                             padded_valid[:, -length:]), atol=1e-7, rtol=1e-6)

    def test_bptt_no_state_leakage_and_batch_independence(self):
        torch.manual_seed(7)
        encoder = self.make_encoder().eval()
        obs = torch.randn(2, 7, 3, requires_grad=True)
        actions = torch.randn(2, 7, 2, requires_grad=True)
        valid = torch.ones(2, 7, dtype=torch.bool)
        valid[0, 2] = False
        first = encoder(obs, actions, valid)
        encoder(torch.randn_like(obs), torch.randn_like(actions), valid)
        torch.testing.assert_close(encoder(obs, actions, valid), first, atol=0, rtol=0)
        changed = obs.detach().clone()
        changed[0] += 100
        torch.testing.assert_close(encoder(changed, actions, valid)[1], first[1], atol=0, rtol=0)
        first.sum().backward()
        self.assertGreater(obs.grad[1, 0].abs().sum().item(), 0)
        self.assertGreater(actions.grad[1, 0].abs().sum().item(), 0)
        self.assertEqual(obs.grad[:, [1, 3, 5]].abs().sum().item(), 0)
        self.assertEqual(obs.grad[0, 2].abs().sum().item(), 0)
        self.assertEqual(actions.grad[:, -1].abs().sum().item(), 0)
        for parameter in encoder.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_initialization_and_input_contract(self):
        encoder = self.make_encoder()
        for projection in (encoder.input_projection, encoder.output_projection):
            bound = math.sqrt(6 / (projection.in_features + projection.out_features))
            self.assertLessEqual(projection.weight.abs().max().item(), bound)
            self.assertEqual(projection.bias.abs().sum().item(), 0)
        for cell in encoder.cells:
            for parameter in cell.parameters():
                self.assertLessEqual(parameter.abs().max().item(), 1 / math.sqrt(cell.hidden_size))
            self.assertGreater(cell.bias_ih.abs().sum().item(), 0)
            self.assertGreater(cell.bias_hh.abs().sum().item(), 0)
        for field in ('obs_dim', 'exec_dim', 'history_length', 'gru_input_dim',
                      'gru_hidden_dim', 'gru_num_layers', 'sub_ratio'):
            for value in (0, -1, 1.5, True):
                args = dict(obs_dim=3, exec_dim=2, history_length=7)
                args[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(AssertionError):
                    GRUHistoryEncoder(**args)
        obs, actions, valid = torch.randn(2, 7, 3), torch.randn(2, 7, 2), torch.ones(2, 7, dtype=torch.bool)
        for bad_valid in (valid.float(), torch.zeros_like(valid), valid.clone()):
            if bad_valid.dtype == torch.bool:
                bad_valid[:, -1] = False
            with self.assertRaises(AssertionError):
                encoder(obs, actions, bad_valid)
        with self.assertRaises(AssertionError):
            encoder(obs, actions.double(), valid)
        with self.assertRaises(AssertionError):
            encoder(obs.long(), actions.long(), valid)

    def test_optimizer_ownership_and_target_updates_include_all_gru_parameters(self):
        policy = policy_tests.ReactiveHistoryPolicyTest().make_policy()
        optimizers = policy.get_optimizer(1e-4, 1e-4)
        for model, optimizer in ((policy.actor, optimizers['actor_optimizer']),
                                  (policy.qs, optimizers['q_optimizer'])):
            owned = {id(p) for group in optimizer.param_groups for p in group['params']}
            self.assertTrue(all(id(p) in owned for p in model.encoder.parameters()))
        before = {name: p.clone() for name, p in policy.q_targets.encoder.named_parameters()}
        with torch.no_grad():
            for parameter in policy.qs.encoder.parameters():
                parameter.add_(0.1)
        policy.target_update()
        online = dict(policy.qs.encoder.named_parameters())
        for name, parameter in policy.q_targets.encoder.named_parameters():
            torch.testing.assert_close(parameter, before[name] * (1 - policy.tau) + online[name] * policy.tau)
            self.assertFalse(parameter.requires_grad)
        self.assertFalse(policy.q_targets.training)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_packed_gru_equivalence_cuda_float32(self):
        for ratio in (1, 2):
            self.compare_packed(ratio, torch.float32, 'cuda', 2e-4, 2e-6)


if __name__ == '__main__':
    unittest.main()
