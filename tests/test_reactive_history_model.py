import copy
import inspect
import unittest

import torch

from zprl.model.online import BatchedLinear
from zprl.model.rl_history import HistoryActor, HistoryEncoder, HistoryQEnsemble


class ReactiveHistoryModelTest(unittest.TestCase):
    def make_inputs(self, history_length=4, batch_size=2):
        obs = torch.randn(batch_size, history_length, 3)
        a_exec = torch.randn(batch_size, history_length, 2)
        valid = torch.ones(batch_size, history_length, dtype=torch.bool)
        base = torch.randn(batch_size, 2)
        return obs, a_exec, valid, base

    def make_models(self, history_length=4, causal_mask=False, dropout=0.0):
        common = dict(obs_dim=3, exec_dim=2, history_length=history_length,
                      hidden_dim=8, d_model=8, nhead=2, num_layers=2,
                      dim_feedforward=16, dropout=dropout, causal_mask=causal_mask)
        return (HistoryActor(action_dim=2, **common),
                HistoryQEnsemble(num_qs=3, **common))

    def test_model_interface_and_three_hidden_layer_heads(self):
        actor, qs = self.make_models()
        self.assertTrue(all(layer.norm_first for layer in actor.encoder.transformer.layers))
        self.assertTrue(all(layer.norm_first for layer in qs.encoder.transformer.layers))
        self.assertEqual(actor.encoder.projection.in_features, 5)
        self.assertEqual(qs.encoder.projection.in_features, 5)
        self.assertNotIn('exec_steps', inspect.signature(HistoryEncoder.forward).parameters)
        self.assertEqual(len([m for m in actor.net if isinstance(m, torch.nn.Linear)]), 3)
        self.assertEqual(len([m for m in qs.net if isinstance(m, BatchedLinear)]), 4)
        self.assertEqual(qs.net[0].weight.shape, (3, 10, 8))
        self.assertEqual(sum(isinstance(m, HistoryEncoder) for m in qs.modules()), 1)
        self.assertIsNot(actor.encoder, qs.encoder)
        self.assertNotEqual(actor.encoder.projection.weight.data_ptr(),
                            qs.encoder.projection.weight.data_ptr())
        self.assertNotEqual(qs.net[0].weight[0].data_ptr(),
                            qs.net[0].weight[1].data_ptr())

    def test_parameter_log_separates_actor_and_critic_modules(self):
        with self.assertLogs('zprl.model.rl_history', level='INFO') as captured:
            actor, qs = self.make_models()
        actor_log, critic_log = captured.output
        self.assertIn('HistoryActor parameters:', actor_log)
        self.assertIn('HistoryQEnsemble parameters:', critic_log)
        for module, message in ((actor, actor_log), (qs, critic_log)):
            for name in ('encoder', 'net'):
                count = sum(p.numel() for p in getattr(module, name).parameters()) / 1e6
                self.assertIn(f'{name}={count:.4f} M', message)
            self.assertNotIn('total=', message)
            self.assertNotIn('q_head_each=', message)

    def test_shapes_sampling_and_gradients(self):
        for history_length, causal_mask in ((1, False), (1, True), (4, False), (4, True)):
            with self.subTest(history_length=history_length, causal_mask=causal_mask):
                actor, qs = self.make_models(history_length, causal_mask)
                obs, a_exec, valid, base = self.make_inputs(history_length)
                mean, logstd = actor(obs, a_exec, valid, base)
                self.assertEqual(mean.shape, (2, 2))
                self.assertEqual(logstd.shape, (2, 2))
                result = actor.get_action(obs, a_exec, valid, base)
                self.assertEqual({k: v.shape for k, v in result.items()},
                                 {'sample': (2, 2), 'mean': (2, 2),
                                  'log_prob': (2, 1), 'log_std': (2, 2)})
                torch.testing.assert_close(actor.get_eval_action(obs, a_exec, valid, base),
                                           mean.tanh())
                candidate = torch.randn(2, 2, requires_grad=True)
                value = qs(obs, a_exec, valid, candidate)
                self.assertEqual(value.shape, (3, 2, 1))
                (result['sample'].sum() + value.sum()).backward()
                self.assertIsNotNone(actor.encoder.projection.weight.grad)
                self.assertIsNotNone(qs.encoder.projection.weight.grad)
                self.assertIsNotNone(qs.net[0].weight.grad)
                self.assertGreater(candidate.grad.abs().sum().item(), 0)

    def test_padding_and_current_action_cannot_change_output(self):
        for causal_mask in (False, True):
            with self.subTest(causal_mask=causal_mask):
                actor, qs = self.make_models(causal_mask=causal_mask)
                actor.eval()
                qs.eval()
                obs, a_exec, valid, base = self.make_inputs()
                valid[0, :3] = False
                valid[1, :1] = False
                original_actor = actor(obs, a_exec, valid, base)
                original_q = qs(obs, a_exec, valid, base)
                changed_obs = obs.clone()
                changed_exec = a_exec.clone()
                changed_obs[~valid] = float('nan')
                changed_exec[~valid] = float('nan')
                changed_exec[:, -1] = 1000
                for original, changed in zip(original_actor,
                                             actor(changed_obs, changed_exec, valid, base)):
                    torch.testing.assert_close(original, changed)
                    self.assertTrue(torch.isfinite(changed).all())
                torch.testing.assert_close(original_q,
                                           qs(changed_obs, changed_exec, valid, base))
                self.assertTrue(torch.isfinite(original_q).all())
                actor.encoder.debug = True
                valid[:, -1] = False
                with self.assertRaises(AssertionError):
                    actor(obs, a_exec, valid, base)

    def test_order_and_past_actions_affect_current_representation(self):
        for causal_mask in (False, True):
            with self.subTest(causal_mask=causal_mask):
                encoder = self.make_models(causal_mask=causal_mask)[0].encoder.eval()
                obs, a_exec, valid, _ = self.make_inputs()
                original = encoder(obs, a_exec, valid)
                changed_exec = a_exec.clone()
                changed_exec[:, 0] += 5
                self.assertGreater((encoder(obs, changed_exec, valid) - original).abs().max(), 1e-6)
                self.assertGreater((encoder(obs.flip(1), a_exec.flip(1), valid) - original).abs().max(), 1e-6)

    def test_causal_mode_changes_multilayer_encoding(self):
        actor, _ = self.make_models(causal_mask=False)
        causal, _ = self.make_models(causal_mask=True)
        causal.encoder.load_state_dict(actor.encoder.state_dict())
        obs, a_exec, valid, _ = self.make_inputs()
        self.assertGreater((actor.encoder.eval()(obs, a_exec, valid) -
                            causal.encoder.eval()(obs, a_exec, valid)).abs().max(), 1e-6)

    def test_target_copy_and_dtype(self):
        _, qs = self.make_models()
        target = copy.deepcopy(qs).requires_grad_(False).eval()
        self.assertEqual(len(list(qs.parameters())), len(list(target.parameters())))
        self.assertTrue(all(not p.requires_grad for p in target.parameters()))
        self.assertNotEqual(qs.encoder.projection.weight.data_ptr(),
                            target.encoder.projection.weight.data_ptr())
        obs, a_exec, valid, base = self.make_inputs()
        qs.eval()
        torch.testing.assert_close(qs(obs, a_exec, valid, base),
                                   target(obs, a_exec, valid, base))
        double_qs = qs.double()
        self.assertEqual(double_qs(obs.double(), a_exec.double(), valid,
                                   base.double()).dtype, torch.float64)

    def test_dropout_follows_train_eval_mode(self):
        actor, _ = self.make_models(dropout=0.5)
        obs, a_exec, valid, base = self.make_inputs()
        actor.eval()
        torch.testing.assert_close(actor(obs, a_exec, valid, base)[0],
                                   actor(obs, a_exec, valid, base)[0])
        actor.train()
        outputs = [actor(obs, a_exec, valid, base)[0] for _ in range(3)]
        self.assertTrue(all(torch.isfinite(output).all() for output in outputs))
        self.assertGreater((outputs[0] - outputs[1]).abs().max(), 1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_mask_and_gradient(self):
        obs, a_exec, valid, base = self.make_inputs()
        valid[0, :2] = False
        for causal_mask in (False, True):
            with self.subTest(causal_mask=causal_mask):
                actor, qs = self.make_models(causal_mask=causal_mask)
                actor, qs = actor.cuda(), qs.cuda()
                cuda_inputs = [x.cuda() for x in (obs, a_exec, valid, base)]
                action = actor.get_action(*cuda_inputs)
                candidate = base.cuda().detach().requires_grad_(True)
                value = qs(*cuda_inputs[:3], candidate)
                self.assertEqual(action['sample'].device.type, 'cuda')
                self.assertEqual(value.device.type, 'cuda')
                (action['log_prob'].mean() + value.mean()).backward()
                self.assertTrue(torch.isfinite(candidate.grad).all())


if __name__ == '__main__':
    unittest.main()
