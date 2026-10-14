import copy
import inspect
import unittest
from unittest.mock import patch

import torch

from zprl.model.gru_history_encoder import GRUHistoryEncoder
from zprl.model.online import BatchedLinear
from zprl.model.rl_history import HistoryActor, HistoryQEnsemble
from zprl.model.transformer import HistoryEncoder, OutputGatedTransformerEncoder


class ReactiveHistoryModelTest(unittest.TestCase):
    def make_inputs(self, history_length=4, batch_size=2):
        obs = torch.randn(batch_size, history_length, 3)
        a_exec = torch.randn(batch_size, history_length, 2)
        valid = torch.ones(batch_size, history_length, dtype=torch.bool)
        base = torch.randn(batch_size, 2)
        return obs, a_exec, valid, base

    def make_models(self, history_length=4, sub_ratio=1):
        common = dict(obs_dim=3, exec_dim=2, history_length=history_length,
                      hidden_dim=8, gru_input_dim=8, gru_hidden_dim=8,
                      gru_num_layers=2, sub_ratio=sub_ratio)
        return (HistoryActor(action_dim=2, **common),
                HistoryQEnsemble(num_qs=3, **common))

    def test_model_interface_and_three_hidden_layer_heads(self):
        actor, qs = self.make_models()
        for encoder in (actor.encoder, qs.encoder):
            self.assertIsInstance(encoder, GRUHistoryEncoder)
            self.assertEqual(len(encoder.cells), 2)
            self.assertEqual(encoder.input_projection.in_features, 5)
            self.assertEqual(encoder.output_dim, 8)
        self.assertNotIn('exec_steps', inspect.signature(GRUHistoryEncoder.forward).parameters)
        self.assertEqual(len([m for m in actor.net if isinstance(m, torch.nn.Linear)]), 3)
        self.assertEqual(len([m for m in qs.net if isinstance(m, BatchedLinear)]), 4)
        self.assertEqual(actor.net[0].in_features, 13)
        self.assertEqual(qs.net[0].weight.shape, (3, 13, 8))
        self.assertEqual(sum(isinstance(m, GRUHistoryEncoder) for m in qs.modules()), 1)
        self.assertIsNot(actor.encoder, qs.encoder)
        self.assertNotEqual(actor.encoder.input_projection.weight.data_ptr(),
                            qs.encoder.input_projection.weight.data_ptr())
        self.assertNotEqual(qs.net[0].weight[0].data_ptr(), qs.net[0].weight[1].data_ptr())

    def test_parameter_log_separates_actor_and_critic_modules(self):
        with self.assertLogs('zprl.model.rl_history', level='INFO') as captured:
            actor, qs = self.make_models()
        for module, message, name in ((actor, captured.output[0], 'HistoryActor'),
                                      (qs, captured.output[1], 'HistoryQEnsemble')):
            self.assertIn(name + ' parameters:', message)
            for part in ('encoder', 'net'):
                count = sum(p.numel() for p in getattr(module, part).parameters()) / 1e6
                self.assertIn(f'{part}={count:.4f} M', message)
            self.assertNotIn('total=', message)
            self.assertNotIn('q_head_each=', message)

    def test_current_observation_reaches_heads_without_gru_features(self):
        for length, ratio in ((1, 1), (4, 2)):
            for mode in ('exec', 'base_res'):
                common = dict(obs_dim=3, exec_dim=2, history_length=length,
                              sub_ratio=ratio, hidden_dim=8, gru_input_dim=8,
                              gru_hidden_dim=8, gru_num_layers=2)
                models = (HistoryActor(action_dim=2, **common).eval(),
                          HistoryQEnsemble(num_qs=3, critic_action_input=mode, **common).eval())
                for model in models:
                    with self.subTest(length=length, ratio=ratio, mode=mode, model=type(model)):
                        obs, a_exec, valid, base = self.make_inputs(length)
                        obs.requires_grad_()
                        residual = torch.randn_like(base, requires_grad=True)
                        candidate = (base, residual) if isinstance(model, HistoryQEnsemble) \
                            and mode == 'base_res' else (base,)
                        captured = []
                        hook = model.net[0].register_forward_pre_hook(
                            lambda module, args: captured.append(args[0]))
                        with patch.object(model.encoder, 'forward', return_value=torch.zeros(2, 8)):
                            result = model(obs, a_exec, valid, *candidate)
                        hook.remove()
                        expected = torch.cat([torch.zeros(2, 8), obs[:, -1], *candidate], dim=-1)
                        if isinstance(model, HistoryQEnsemble):
                            expected = expected[None].expand(3, -1, -1)
                            loss = result.sum()
                        else:
                            loss = result[0].sum()
                        torch.testing.assert_close(captured[0], expected, atol=0, rtol=0)
                        loss.backward()
                        self.assertGreater(obs.grad[:, -1].abs().sum().item(), 0)
                        torch.testing.assert_close(obs.grad[:, :-1], torch.zeros_like(obs.grad[:, :-1]))

    def test_shapes_sampling_and_gradients(self):
        for length in (1, 3, 7):
            for ratio in (1, 2):
                with self.subTest(length=length, ratio=ratio):
                    actor, qs = self.make_models(length, ratio)
                    obs, actions, valid, base = self.make_inputs(length)
                    mean, logstd = actor(obs, actions, valid, base)
                    self.assertEqual(mean.shape, (2, 2))
                    self.assertEqual(logstd.shape, (2, 2))
                    result = actor.get_action(obs, actions, valid, base)
                    self.assertEqual({k: v.shape for k, v in result.items()},
                                     {'sample': (2, 2), 'mean': (2, 2),
                                      'log_prob': (2, 1), 'log_std': (2, 2)})
                    torch.testing.assert_close(actor.get_eval_action(obs, actions, valid, base), mean.tanh())
                    candidate = torch.randn(2, 2, requires_grad=True)
                    value = qs(obs, actions, valid, candidate)
                    self.assertEqual(value.shape, (3, 2, 1))
                    (result['sample'].sum() + value.sum()).backward()
                    for encoder in (actor.encoder, qs.encoder):
                        self.assertGreater(encoder.input_projection.weight.grad.abs().sum().item(), 0)
                        self.assertGreater(encoder.output_projection.weight.grad.abs().sum().item(), 0)
                        self.assertTrue(all(cell.weight_ih.grad is not None for cell in encoder.cells))
                    self.assertGreater(candidate.grad.abs().sum().item(), 0)

    def test_padding_and_current_action_cannot_change_output(self):
        actor, qs = self.make_models()
        actor.eval()
        qs.eval()
        obs, actions, valid, base = self.make_inputs()
        valid[0, :3] = False
        valid[1, :1] = False
        original_actor = actor(obs, actions, valid, base)
        original_q = qs(obs, actions, valid, base)
        changed_obs, changed_actions = obs.clone(), actions.clone()
        changed_obs[~valid] = float('nan')
        changed_actions[~valid] = float('nan')
        changed_actions[:, -1] = float('nan')
        torch.testing.assert_close(actor(changed_obs, changed_actions, valid, base), original_actor)
        torch.testing.assert_close(qs(changed_obs, changed_actions, valid, base), original_q)
        actor.encoder.debug = True
        for bad_mask in (valid & False, valid.clone()):
            bad_mask[:, -1] = False
            with self.assertRaises(AssertionError):
                actor(obs, actions, bad_mask, base)

    def test_order_and_past_actions_affect_current_representation(self):
        encoder = self.make_models()[0].encoder.eval()
        obs, actions, valid, _ = self.make_inputs()
        original = encoder(obs, actions, valid)
        changed = actions.clone()
        changed[:, 0] += 5
        self.assertGreater((encoder(obs, changed, valid) - original).abs().max(), 1e-6)
        self.assertGreater((encoder(obs.flip(1), actions.flip(1), valid) - original).abs().max(), 1e-6)

    def test_target_copy_and_dtype(self):
        _, qs = self.make_models()
        target = copy.deepcopy(qs).requires_grad_(False).eval()
        self.assertEqual(len(list(qs.parameters())), len(list(target.parameters())))
        self.assertTrue(all(not p.requires_grad for p in target.parameters()))
        self.assertNotEqual(qs.encoder.input_projection.weight.data_ptr(),
                            target.encoder.input_projection.weight.data_ptr())
        obs, actions, valid, base = self.make_inputs()
        qs.eval()
        torch.testing.assert_close(qs(obs, actions, valid, base), target(obs, actions, valid, base))
        qs.double()
        self.assertEqual(qs(obs.double(), actions.double(), valid, base.double()).dtype, torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_mask_and_gradient(self):
        actor, qs = self.make_models()
        actor, qs = actor.cuda(), qs.cuda()
        inputs = [value.cuda() for value in self.make_inputs()]
        inputs[2][0, :-1] = False
        action = actor.get_action(*inputs)
        candidate = inputs[3].detach().requires_grad_(True)
        value = qs(*inputs[:3], candidate)
        (action['log_prob'].mean() + value.mean()).backward()
        self.assertTrue(torch.isfinite(candidate.grad).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))


class TransformerHistoryEncoderTest(unittest.TestCase):
    def make_encoder(self, causal=False, dropout=0):
        return HistoryEncoder(3, 2, 4, d_model=8, nhead=2, num_layers=2,
                              dim_feedforward=16, dropout=dropout, causal_mask=causal, sub_ratio=1)

    def test_padding_and_causal_behavior_are_preserved(self):
        for causal in (False, True):
            encoder = self.make_encoder(causal).eval()
            self.assertIsInstance(encoder.transformer, OutputGatedTransformerEncoder)
            obs, actions, valid, _ = ReactiveHistoryModelTest().make_inputs()
            valid[0, :-1] = False
            expected = encoder(obs, actions, valid)
            obs[~valid] = float('nan')
            actions[~valid] = float('nan')
            actions[:, -1] = float('nan')
            torch.testing.assert_close(encoder(obs, actions, valid), expected)
            self.assertEqual(expected.shape, (2, 11))
        ordinary, causal = self.make_encoder().eval(), self.make_encoder(True).eval()
        causal.load_state_dict(ordinary.state_dict())
        obs, actions, valid, _ = ReactiveHistoryModelTest().make_inputs()
        self.assertGreater((ordinary(obs, actions, valid) - causal(obs, actions, valid)).abs().max(), 1e-6)

    def test_dropout_follows_train_eval_mode(self):
        encoder = self.make_encoder(dropout=0.5)
        obs, actions, valid, _ = ReactiveHistoryModelTest().make_inputs()
        encoder.eval()
        torch.testing.assert_close(encoder(obs, actions, valid), encoder(obs, actions, valid))
        encoder.train()
        outputs = [encoder(obs, actions, valid) for _ in range(3)]
        self.assertTrue(all(torch.isfinite(output).all() for output in outputs))
        self.assertGreater((outputs[0] - outputs[1]).abs().max(), 1e-6)


if __name__ == '__main__':
    unittest.main()
