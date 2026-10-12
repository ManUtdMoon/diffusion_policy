import tempfile
import unittest
from pathlib import Path

import dill
import hydra
import torch
from omegaconf import OmegaConf

from tests import test_reactive_history_policy as policy_tests
from tests.test_reactive_history_replay import add_step, make_buffer, sample_at
from zprl.common.reactive_history import HistoryWindow
from zprl.model.rl_history import (
    FlattenHistoryEncoder, HistoryActor, HistoryEncoder, HistoryMLPActor,
    HistoryMLPQEnsemble, HistoryQEnsemble,
)


class ReactiveHistorySubsamplingTest(unittest.TestCase):
    def test_current_anchored_selection_padding_and_gradients(self):
        for length, ratio, indices in ((1, 2, [0]), (2, 2, [1]),
                                       (3, 2, [0, 2]), (4, 2, [1, 3]),
                                       (5, 3, [1, 4]), (3, 8, [2])):
            with self.subTest(length=length, ratio=ratio):
                encoder = FlattenHistoryEncoder(1, 1, length, debug=True, sub_ratio=ratio)
                obs = torch.arange(length, dtype=torch.float32).reshape(1, length, 1)
                actions = obs + 10
                expected = [value for index in indices[:-1] for value in (index, index + 10)]
                expected.append(indices[-1])
                skipped = [index for index in range(length) if index not in indices]
                obs[:, skipped] = float('nan')
                actions[:, skipped] = float('nan')
                actions[:, -1] = float('nan')
                obs.requires_grad_()
                actions.requires_grad_()
                valid = torch.ones(1, length, dtype=torch.bool)
                actual = encoder(obs, actions, valid)
                torch.testing.assert_close(actual, torch.tensor([expected], dtype=torch.float32))
                self.assertEqual(encoder.output_dim, len(expected))
                actual.sum().backward()
                self.assertEqual(obs.grad[:, skipped].abs().sum().item(), 0)
                self.assertEqual(actions.grad[:, skipped + [length - 1]].abs().sum().item(), 0)
                if len(indices) > 1:
                    valid[:, indices[0]] = False
                    expected[:2] = [0, 0]
                    torch.testing.assert_close(encoder(obs, actions, valid),
                                               torch.tensor([expected], dtype=torch.float32))

    def test_actor_and_critic_match_explicitly_selected_history(self):
        for cls in (HistoryMLPActor, HistoryMLPQEnsemble, HistoryActor, HistoryQEnsemble):
            for length, indices in ((3, [0, 2]), (4, [1, 3])):
                with self.subTest(model=cls.__name__, length=length):
                    common = dict(obs_dim=3, exec_dim=2, hidden_dim=8)
                    critic = cls in (HistoryMLPQEnsemble, HistoryQEnsemble)
                    common.update(dict(num_qs=3, critic_action_input='base_res') if critic
                                  else dict(action_dim=2))
                    if cls in (HistoryActor, HistoryQEnsemble):
                        common.update(d_model=8, nhead=2, num_layers=2,
                                      dim_feedforward=16, dropout=0, causal_mask=True)
                    model = cls(history_length=length, sub_ratio=2, **common).eval()
                    reference = cls(history_length=2, **common).eval()
                    reference.load_state_dict(model.state_dict())
                    obs = torch.randn(2, length, 3)
                    actions = torch.randn(2, length, 2)
                    valid = torch.ones(2, length, dtype=torch.bool)
                    valid[0, indices[0]] = False
                    base = torch.randn(2, 2)
                    candidate = (base, torch.randn(2, 2)) if critic else (base,)
                    expected = reference(obs[:, indices], actions[:, indices],
                                         valid[:, indices], *candidate)
                    skipped = [index for index in range(length) if index not in indices]
                    obs[:, skipped] = float('nan')
                    actions[:, skipped] = float('nan')
                    actual = model(obs, actions, valid, *candidate)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_replay_current_and_n_step_endpoint_match_online_windows(self):
        buffer = make_buffer(n_steps=2, history_length=5)
        window = HistoryWindow(5, 2)
        window.reset(torch.tensor([[0.]]))
        online = {}
        for step in range(10):
            online[step] = window.get()
            action = torch.tensor([[step + 10., step + 10.25]])
            window.complete(action, torch.tensor([2]))
            window.append(torch.tensor([[float(step + 1)]]))
            add_step(buffer, step, 0, 0, step + 10, 0, step + 1, 0, 0)
        online[10] = window.get()
        encoder = FlattenHistoryEncoder(1, 2, 5, sub_ratio=2)
        for index in (0, 1, 6, 8):
            batch = sample_at(buffer, [index])
            torch.testing.assert_close(encoder(batch.obs, batch.a_exec, batch.obs_valid),
                                       encoder(*online[index][:3]))
            torch.testing.assert_close(encoder(batch.next_obs, batch.next_a_exec,
                                               batch.next_obs_valid),
                                       encoder(*online[index + 2][:3]))

    def test_checkpoint_retains_ratio_and_losses_backpropagate(self):
        cfg = OmegaConf.create(dict(
            _target_='zprl.policy.reactive_history_policy.HistoryResiduePolicy',
            obs_dim=3, action_dim=2, history_length=3, sub_ratio=2,
            hidden_dim=8, num_qs=3, critic_action_input='base_res'))
        policy = hydra.utils.instantiate(cfg).eval()
        batch = policy_tests.ReactiveHistoryPolicyTest().make_batch()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'online.ckpt'
            torch.save({'cfg': cfg, 'res_policy': policy.state_dict()}, checkpoint, pickle_module=dill)
            payload = torch.load(checkpoint, pickle_module=dill)
            restored = hydra.utils.instantiate(payload['cfg']).eval()
            restored.load_state_dict(payload['res_policy'])
            self.assertEqual(restored.sub_ratio, 2)
            for method in ('compute_critic_loss', 'compute_actor_loss', 'compute_alpha_loss'):
                torch.manual_seed(7)
                expected = getattr(policy, method)(batch)
                torch.manual_seed(7)
                actual = getattr(restored, method)(batch)
                expected = expected[0] if isinstance(expected, tuple) else expected
                actual = actual[0] if isinstance(actual, tuple) else actual
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                restored.zero_grad(set_to_none=True)
                actual.backward()
                self.assertTrue(torch.isfinite(actual))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in restored.parameters()
                                    if p.grad is not None))

    def test_invalid_ratio_is_rejected(self):
        for encoder in (FlattenHistoryEncoder, HistoryEncoder):
            for ratio in (0, -1, 1.5):
                with self.subTest(encoder=encoder.__name__, ratio=ratio):
                    with self.assertRaises(AssertionError):
                        encoder(3, 2, 3, sub_ratio=ratio)


if __name__ == '__main__':
    unittest.main()
