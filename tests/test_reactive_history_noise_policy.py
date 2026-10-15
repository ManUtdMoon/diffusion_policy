import copy
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from zprl.common.reactive_history import NoiseHistoryReplayBufferSamples
from zprl.model.gru_history_encoder import GRUHistoryEncoder
from zprl.model.common.normalizer import SingleFieldLinearNormalizer
from zprl.model.rl_history import HistoryActor, HistoryQEnsemble
from zprl.policy.flow_match_unet_image_policy import FlowMatchUnetImagePolicy
from zprl.policy.reactive_history_noise_policy import HistoryNoisePolicy, HistoryNoiseSumPolicy


def make_policy(action_dim=10, exec_dim=40, history_length=3, **kwargs):
    return HistoryNoisePolicy(
        obs_dim=3, action_dim=action_dim, exec_dim=exec_dim,
        history_length=history_length, hidden_dim=8, num_qs=3, num_subset=2,
        gru_input_dim=8, gru_hidden_dim=8, gru_num_layers=2, sub_ratio=1, **kwargs)


def make_batch(policy):
    bs, length = 2, policy.history_length
    obs = torch.randn(bs, length, policy.obs_dim)
    valid = torch.ones(bs, length, dtype=torch.bool)
    valid[0, :-1] = False
    return NoiseHistoryReplayBufferSamples(
        obs=obs, a_exec=torch.randn(bs, length, policy.exec_dim), obs_valid=valid,
        exec_steps=torch.zeros(bs, length, dtype=torch.long),
        next_obs=torch.randn_like(obs), next_a_exec=torch.randn(bs, length, policy.exec_dim),
        next_obs_valid=valid.clone(), next_exec_steps=torch.zeros(bs, length, dtype=torch.long),
        action=torch.rand(bs, policy.action_dim) * 2 - 1,
        rewards=torch.tensor([[1.], [2.]]), dones=torch.tensor([[0.], [1.]]),
        discounts=torch.tensor([[0.8], [0.6]]))


class HistoryNoiseModelTest(unittest.TestCase):
    def test_independent_dimensions_and_noise_gradient(self):
        for noise_dim, exec_dim in ((10, 40), (20, 10)):
            policy = make_policy(noise_dim, exec_dim)
            batch = make_batch(policy)
            actor, critic = policy.actor, policy.qs
            self.assertIsInstance(actor, HistoryActor)
            self.assertIsInstance(critic, HistoryQEnsemble)
            self.assertEqual(actor.net[0].in_features, 8 + 3)
            self.assertEqual(critic.net[0].weight.shape[1], 8 + 3 + noise_dim)
            candidate = batch.action.detach().requires_grad_(True)
            sample = actor.get_action(batch.obs, batch.a_exec, batch.obs_valid)
            self.assertEqual(sample['sample'].shape, (2, noise_dim))
            self.assertEqual(sample['log_prob'].shape, (2, 1))
            values = critic(batch.obs, batch.a_exec, batch.obs_valid, candidate)
            self.assertEqual(values.shape, (3, 2, 1))
            (values.sum() + sample['sample'].sum()).backward()
            self.assertGreater(candidate.grad.abs().sum().item(), 0)
            for encoder in (actor.encoder, critic.encoder):
                for cell in encoder.cells:
                    self.assertGreater(cell.weight_ih.grad.abs().sum().item(), 0)
            with self.assertRaises(AssertionError):
                actor(batch.obs, batch.a_exec, batch.obs_valid, torch.zeros(2, exec_dim))
            with self.assertRaises(AssertionError):
                critic(batch.obs, batch.a_exec, batch.obs_valid, torch.zeros(2, exec_dim))

    def test_current_obs_direct_path_and_h1_gru(self):
        policy = make_policy(history_length=1)
        batch = make_batch(policy)
        self.assertIsInstance(policy.actor.encoder, GRUHistoryEncoder)
        expected = policy.predict_noise(batch.obs, batch.a_exec, batch.obs_valid, argmax=True)
        changed = torch.full_like(batch.a_exec, float('nan'))
        torch.testing.assert_close(
            policy.predict_noise(batch.obs, changed, batch.obs_valid, argmax=True), expected)
        captured = []
        hook = policy.actor.net[0].register_forward_pre_hook(
            lambda module, args: captured.append(args[0]))
        obs = batch.obs.detach().requires_grad_(True)
        with patch.object(policy.actor.encoder, 'forward', return_value=torch.zeros(2, 8)):
            mean, _ = policy.actor(obs, batch.a_exec, batch.obs_valid)
        hook.remove()
        torch.testing.assert_close(captured[0], torch.cat([torch.zeros(2, 8), obs[:, -1]], -1))
        mean.sum().backward()
        self.assertGreater(obs.grad.abs().sum().item(), 0)

    def test_invalid_history_and_current_action_do_not_leak(self):
        policy = make_policy()
        batch = make_batch(policy)
        original = policy.actor(batch.obs, batch.a_exec, batch.obs_valid)
        original_q = policy.qs(batch.obs, batch.a_exec, batch.obs_valid, batch.action)
        changed_obs, changed_exec = batch.obs.clone(), batch.a_exec.clone()
        changed_obs[~batch.obs_valid] = float('nan')
        changed_exec[~batch.obs_valid] = float('nan')
        changed_exec[:, -1] = float('nan')
        torch.testing.assert_close(policy.actor(changed_obs, changed_exec, batch.obs_valid), original)
        torch.testing.assert_close(policy.qs(changed_obs, changed_exec, batch.obs_valid, batch.action), original_q)


class DummyObsEncoder(torch.nn.Module):
    def output_shape(self):
        return (3,)


class UnetNoiseDecoderTest(unittest.TestCase):
    def test_supplied_noise_matches_original_flow_integration_and_alignment(self):
        base = FlowMatchUnetImagePolicy(
            shape_meta={'action': {'shape': [2]}}, obs_encoder=DummyObsEncoder(),
            horizon=8, n_action_steps=3, n_obs_steps=2, num_inference_steps=3,
            diffusion_step_embed_dim=8, down_dims=(8, 16), n_groups=2).eval()
        base.normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
        noise = torch.randn(2, 8, 2)
        original_noise = noise.clone()
        cond = torch.randn(2, 6)
        with patch('torch.randn', return_value=noise.clone()):
            original = base.conditional_sample(torch.zeros_like(noise),
                                               torch.zeros_like(noise, dtype=torch.bool),
                                               global_cond=cond)
        result = base.conditional_predict_from_noise(cond, noise)
        torch.testing.assert_close(noise, original_noise, atol=0, rtol=0)
        torch.testing.assert_close(result['naction_pred'], original, atol=0, rtol=0)
        torch.testing.assert_close(result['naction'], original[:, 1:4], atol=0, rtol=0)
        torch.testing.assert_close(result['action'], original[:, 1:4], atol=0, rtol=0)
        torch.testing.assert_close(base.conditional_predict_from_noise(cond, noise.double())['naction_pred'], original)
        with self.assertRaises(AssertionError):
            base.conditional_predict_from_noise(cond, noise[:, :4])
        with self.assertRaises(AssertionError):
            base.conditional_predict_from_noise(cond[:, :3], noise)


class HistoryNoiseLossTest(unittest.TestCase):
    def test_optimizer_ownership_and_real_updates(self):
        policy = make_policy(tau=0.2)
        batch = make_batch(policy)
        for enc_lr in (None, 0, 1e-5):
            optimizers = policy.get_optimizer(2e-4, 3e-4, enc_lr)
            for key, module, head_lr in (('actor_optimizer', policy.actor, 2e-4),
                                         ('q_optimizer', policy.qs, 3e-4)):
                groups = optimizers[key].param_groups
                ids = [id(p) for group in groups for p in group['params']]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(set(ids), {id(p) for p in module.parameters()})
                self.assertEqual(groups[0]['lr'], head_lr)
                if enc_lr is not None:
                    self.assertEqual(groups[1]['lr'], enc_lr)
                    self.assertEqual({id(p) for p in groups[1]['params']},
                                     {id(p) for p in module.encoder.parameters()})
        optimizers = policy.get_optimizer(1e-2, 1e-2, 1e-3)
        for method, key, module in (('compute_critic_loss', 'q_optimizer', policy.qs),
                                     ('compute_actor_loss', 'actor_optimizer', policy.actor)):
            policy.zero_grad()
            before = [p.detach().clone() for p in module.encoder.parameters()]
            loss, info = getattr(policy, method)(batch)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNone(policy.log_alpha.grad)
            self.assertTrue(all(p.grad is None for p in policy.q_targets.parameters()))
            other = policy.actor if module is policy.qs else policy.qs
            self.assertTrue(all(p.grad is None for p in other.parameters()))
            for cell in module.encoder.cells:
                self.assertGreater(cell.weight_ih.grad.abs().sum().item(), 0)
            self.assertTrue(all(isinstance(value, torch.Tensor) and not value.requires_grad
                                for value in info.values()))
            optimizers[key].step()
            self.assertTrue(any(not torch.equal(p, old) for p, old in
                                zip(module.encoder.parameters(), before)))
        policy.zero_grad()
        alpha = policy.log_alpha.detach().clone()
        policy.compute_alpha_loss(batch).backward()
        self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))
        optimizers['alpha_optimizer'].step()
        self.assertFalse(torch.equal(alpha, policy.log_alpha))
        targets = [p.clone() for p in policy.q_targets.parameters()]
        policy.target_update()
        for online, target, old in zip(policy.qs.parameters(), policy.q_targets.parameters(), targets):
            torch.testing.assert_close(target, old.lerp(online, 0.2))
        policy.train()
        self.assertFalse(policy.q_targets.training)

    def test_q_action_gradient_survives_freeze_and_modes_restore(self):
        policy = make_policy(auto_alpha=False)
        policy.init_alpha = 0
        batch = make_batch(policy)
        policy.actor.eval()
        policy.qs.train()
        loss, _ = policy.compute_actor_loss(batch)
        self.assertFalse(policy.actor.training)
        self.assertTrue(policy.qs.training)
        loss.backward()
        self.assertGreater(policy.actor.mean.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in policy.qs.parameters()))
        self.assertTrue(all(p.requires_grad for p in policy.qs.parameters()))

    def test_n_step_endpoint_noise_subset_entropy_clip_and_reduction(self):
        policy = make_policy(auto_alpha=False)
        policy.init_alpha = 0.25
        policy.num_subset = policy.num_qs
        batch = make_batch(policy)
        values = torch.tensor([[[3.], [5.]], [[4.], [6.]], [[7.], [8.]]])
        noise = torch.zeros(2, policy.action_dim)
        for q_ent, clip, expected in ((False, None, [3.4, 2.]),
                                      (True, None, [3.8, 2.]),
                                      (True, [0., 1.], [1., 1.])):
            policy.q_ent, policy.target_clip = q_ent, clip
            with patch.object(policy, '_sample_noise_log_prob',
                              return_value=(noise, torch.full((2, 1), -2.), None)) as sample, \
                 patch.object(policy.q_targets, 'forward', return_value=values) as target, \
                 patch('zprl.policy.reactive_history_noise_policy.F.mse_loss',
                       wraps=torch.nn.functional.mse_loss) as mse:
                loss, _ = policy.compute_critic_loss(batch)
            for actual, wanted in zip(sample.call_args.args,
                                      (batch.next_obs, batch.next_a_exec, batch.next_obs_valid)):
                self.assertIs(actual, wanted)
            self.assertIs(target.call_args.args[-1], noise)
            predictions, targets = mse.call_args.args[:2]
            torch.testing.assert_close(targets, torch.tensor(expected)[None].expand(3, -1))
            torch.testing.assert_close(loss, (predictions - targets).square().mean(-1).sum())
        # Check actual subset selection, rather than accidentally averaging all Qs.
        policy.num_subset, policy.q_ent, policy.target_clip = 2, False, None
        with patch.object(policy, '_sample_noise_log_prob', return_value=(noise, torch.zeros(2, 1), None)), \
             patch.object(policy.q_targets, 'forward', return_value=values), \
             patch('torch.randperm', return_value=torch.tensor([2, 1, 0])):
            _, info = policy.compute_critic_loss(batch)
        torch.testing.assert_close(info['q_target'], torch.tensor(3.1))
        no_discounts = batch._replace(discounts=None)
        with patch.object(policy, '_sample_noise_log_prob', return_value=(noise, torch.zeros(2, 1), None)), \
             patch.object(policy.q_targets, 'forward', return_value=values), \
             patch('torch.randperm', return_value=torch.tensor([2, 1, 0])):
            _, info = policy.compute_critic_loss(no_discounts)
        torch.testing.assert_close(info['q_target'], torch.tensor((1 + policy.gamma * 4 + 2) / 2))

    def test_explicit_entropy_noise_rms_and_tensor_metrics(self):
        policy = make_policy(target_entropy=1.25)
        batch = make_batch(policy)
        noise = torch.stack([torch.ones(10), torch.zeros(10)])
        log_prob = torch.tensor([[-2.], [-4.]])
        with patch.object(policy, '_sample_noise_log_prob',
                          return_value=(noise, log_prob, torch.zeros_like(noise))), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('eager item')):
            alpha_loss = policy.compute_alpha_loss(batch)
            actor_loss, info = policy.compute_actor_loss(batch)
        torch.testing.assert_close(alpha_loss, -policy.log_alpha * (-3 + 1.25))
        torch.testing.assert_close(info['res_n_rms'], torch.tensor(0.5))
        self.assertEqual(set(info), {'actor_entropy', 'res_n_rms', 'log_std'})
        self.assertTrue(torch.isfinite(actor_loss))
        for clip in ([0.], [0., 1., 2.], [2., -1.]):
            with self.assertRaises(ValueError):
                make_policy(target_clip=clip)

    def test_forward_compile_preserves_noise_and_q(self):
        policy = make_policy().eval()
        batch = make_batch(policy)
        compiled = copy.deepcopy(policy)
        for module in (compiled.actor, compiled.qs, compiled.q_targets):
            module.forward = torch.compile(module.forward, backend='aot_eager', fullgraph=True)
        torch.testing.assert_close(
            compiled.predict_noise(batch.obs, batch.a_exec, batch.obs_valid, argmax=True),
            policy.predict_noise(batch.obs, batch.a_exec, batch.obs_valid, argmax=True))
        for name in ('qs', 'q_targets'):
            torch.testing.assert_close(
                getattr(compiled, name)(batch.obs, batch.a_exec, batch.obs_valid, batch.action),
                getattr(policy, name)(batch.obs, batch.a_exec, batch.obs_valid, batch.action))


class AffineNormalizer:
    def unnormalize(self, action):
        return action * 3 + 5


class FakeNoiseBase(torch.nn.Module):
    def __init__(self, n_obs_steps=1, action_dim=10):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.horizon = 8
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = 4
        self.obs_feature_dim = 1
        self.action_dim = action_dim
        self.normalizer = {'action': AffineNormalizer()}
        self.decode_calls = []

    def encode_obs(self, obs):
        assert 'completed_stage_mask' not in obs
        return obs['state'].reshape(obs['state'].shape[0], -1)

    def conditional_predict_from_noise(self, cond, noise):
        self.decode_calls.append((cond.clone(), noise.clone()))
        plan = noise + torch.arange(self.horizon, device=noise.device)[None, :, None]
        start = self.n_obs_steps - 1
        return {'naction_pred': plan, 'action_pred': self.normalizer['action'].unnormalize(plan),
                'naction': plan[:, start:start + self.n_action_steps],
                'action': self.normalizer['action'].unnormalize(plan[:, start:start + self.n_action_steps])}


def make_sum(n_noise_steps=1, n_action_steps=4, n_obs_steps=1, history_length=3):
    base = FakeNoiseBase(n_obs_steps)
    noise = HistoryNoisePolicy(
        obs_dim=n_obs_steps + 1, action_dim=n_noise_steps * 10,
        exec_dim=n_action_steps * 10, history_length=history_length, hidden_dim=8,
        gru_input_dim=8, gru_hidden_dim=8, gru_num_layers=2, sub_ratio=1)
    combined = HistoryNoiseSumPolicy(
        noise_scale=1.5, base_obs_emb_dim=n_obs_steps, subtask_dim=1,
        action_dim=10, n_action_steps=n_action_steps, horizon=8,
        n_obs_steps=n_obs_steps, base_policy=base, noise_policy=noise,
        n_noise_steps=n_noise_steps)
    return combined, base


def make_obs(values, masks=None, n_obs_steps=1):
    values = torch.as_tensor(values, dtype=torch.float32)
    if masks is None:
        masks = torch.zeros_like(values)
    return {'state': values[:, None, None].repeat(1, n_obs_steps, 1),
            'completed_stage_mask': torch.as_tensor(masks, dtype=torch.float32)[:, None, None].repeat(1, n_obs_steps, 1)}


class HistoryNoiseRolloutTest(unittest.TestCase):
    def test_repeat_scale_full_decode_and_obs_alignment(self):
        for tn, ta, to in ((1, 4, 1), (2, 1, 2)):
            combined, base = make_sum(tn, ta, to)
            combined.reset(make_obs([1, 2], [0, 1], to))
            self.assertEqual(len(base.decode_calls), 0)
            noise = torch.arange(2 * tn * 10, dtype=torch.float32).reshape(2, tn * 10) / 40
            with patch.object(combined.noise_policy, 'predict_noise', return_value=noise):
                result = combined.predict_action(None)
            self.assertEqual(len(base.decode_calls), 1)
            cond, init_noise = base.decode_calls[-1]
            self.assertEqual(cond.shape, (2, to))
            expected_noise = noise.reshape(2, tn, 10).repeat(1, 8 // tn, 1) * 1.5
            torch.testing.assert_close(init_noise, expected_noise)
            self.assertEqual(result['naction_pred'].shape, (2, 8, 10))
            self.assertEqual(result['action'].shape, (2, ta, 10))
            torch.testing.assert_close(result['naction'], result['naction_pred'][:, to - 1:to - 1 + ta])
            torch.testing.assert_close(result['action'], result['naction'] * 3 + 5)
            torch.testing.assert_close(combined.pending['action'], noise)
            combined.on_env_step(make_obs([3, 4], n_obs_steps=to), [False, False],
                                 [{'executed_action_steps': ta}] * 2)
            self.assertEqual(len(base.decode_calls), 1)
            with patch.object(combined.noise_policy, 'predict_noise', return_value=noise):
                combined.predict_action(None)
            self.assertEqual(len(base.decode_calls), 2)
            torch.testing.assert_close(base.decode_calls[-1][0][:, -1], torch.tensor([3., 4.]))

    def test_warmup_never_calls_actor_and_threshold_samples_all_envs(self):
        for history_length in (1, 3):
            combined, base = make_sum(history_length=history_length)
            combined.reset(make_obs([1, 2]))
            torch.manual_seed(12)
            expected = torch.randn(2, 10).clamp(-1.5, 1.5) / 1.5
            torch.manual_seed(12)
            with patch.object(combined.noise_policy, 'predict_noise', side_effect=AssertionError('warmup actor')):
                result = combined.predict_action(None, global_step=9, learning_start=10)
            torch.testing.assert_close(result['nnoise'], expected)
            torch.testing.assert_close(base.decode_calls[-1][1], expected[:, None].repeat(1, 8, 1) * 1.5)
            transition = combined.on_env_step(make_obs([3, 4]), [False, False],
                                               [{'executed_action_steps': 4}] * 2)
            torch.testing.assert_close(transition['action'], expected)
            torch.testing.assert_close(combined.history.get()[1][:, -1], torch.zeros(2, 40))
            with patch.object(combined.noise_policy.actor, 'get_action',
                              wraps=combined.noise_policy.actor.get_action) as actor, \
                 patch.object(combined.noise_policy.actor, 'get_eval_action',
                              side_effect=AssertionError('training must sample')):
                combined.predict_action(None, global_step=10, learning_start=10)
            self.assertEqual(actor.call_count, 1)
            self.assertEqual(actor.call_args.args[0].shape[0], 2)

    def test_timeout_partial_execution_and_done_resets_only_selected_env(self):
        combined, base = make_sum()
        combined.reset(make_obs([1, 2, 3]))
        with patch.object(combined.noise_policy, 'predict_noise', return_value=torch.ones(3, 10)):
            result = combined.predict_action(None)
        terminal = {key: value[0] for key, value in make_obs([99], [1]).items()}
        transition = combined.on_env_step(make_obs([4, 5, 6]), [False, True, True], [
            {'executed_action_steps': 4},
            {'executed_action_steps': 1, 'TimeLimit.truncated': True, 'terminal_observation': terminal},
            {'executed_action_steps': 2},
        ])
        torch.testing.assert_close(transition['next_obs'], torch.tensor([[4., 0.], [99., 1.], [6., 0.]]))
        executed = transition['a_exec'].reshape(3, 4, 10)
        torch.testing.assert_close(executed[1, :1], result['naction'][1, :1])
        torch.testing.assert_close(executed[1, 1:], torch.zeros(3, 10))
        torch.testing.assert_close(executed[2, 2:], torch.zeros(2, 10))
        obs, actions, valid, steps = combined.history.get()
        torch.testing.assert_close(obs[0, -2:, 0], torch.tensor([1., 4.]))
        torch.testing.assert_close(actions[0, -2], transition['a_exec'][0])
        torch.testing.assert_close(valid[1:], torch.tensor([[False, False, True]] * 2))
        torch.testing.assert_close(obs[1:, -1, 0], torch.tensor([5., 6.]))
        torch.testing.assert_close(steps[0], torch.tensor([0, 4, 0]))
        self.assertEqual(len(base.decode_calls), 1)

    def test_lifecycle_base_freeze_and_deterministic_eval(self):
        combined, base = make_sum()
        with self.assertRaisesRegex(RuntimeError, 'reset'):
            combined.predict_action(None)
        combined.train()
        self.assertFalse(base.training)
        self.assertFalse(combined.noise_policy.q_targets.training)
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))
        obs = make_obs([1])
        combined.reset(obs)
        original = combined.predict_action(None, deterministic=True)
        with self.assertRaisesRegex(RuntimeError, 'on_env_step'):
            combined.predict_action(None)
        combined.on_env_step(obs, [False], [{'executed_action_steps': 4}])
        with self.assertRaisesRegex(RuntimeError, 'predict_action'):
            combined.on_env_step(obs, [False], [{'executed_action_steps': 4}])
        combined.eval()
        combined.reset(obs)
        torch.testing.assert_close(combined.predict_action(None, deterministic=True)['action'], original['action'])
        self.assertTrue(all(p.grad is None for p in base.parameters()))

    def test_train_and_eval_sample_noise_by_default(self):
        combined, _ = make_sum()
        obs = make_obs([1, 2])
        for training in (True, False):
            combined.train(training).reset(obs)
            with patch.object(combined.noise_policy, 'predict_noise',
                              return_value=torch.zeros(2, 10)) as noise:
                combined.predict_action(None)
            self.assertFalse(noise.call_args.kwargs['argmax'])

    def test_dimension_contracts_reject_incompatible_base_and_policy(self):
        combined, base = make_sum()
        args = dict(noise_scale=1.5, base_obs_emb_dim=1, subtask_dim=1,
                    action_dim=10, n_action_steps=4, horizon=8, n_obs_steps=1,
                    base_policy=base, noise_policy=combined.noise_policy)
        for invalid in ({'horizon': 16}, {'n_obs_steps': 2}, {'action_dim': 7},
                        {'n_action_steps': 0}, {'n_action_steps': 9}, {'n_noise_steps': 3},
                        {'noise_scale': 0}, {'base_obs_emb_dim': 2}, {'n_noise_steps': 2},
                        {'n_action_steps': 1}):
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                HistoryNoiseSumPolicy(**{**args, **invalid})

    def test_actual_state_dict_save_load_round_trip(self):
        combined, _ = make_sum()
        obs = make_obs([1, 2])
        combined.eval().reset(obs)
        expected = combined.predict_action(None, deterministic=True)['action']
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'noise.ckpt'
            torch.save({'cfg': OmegaConf.create({'noise_policy': {'history_length': 3}}),
                        'noise_policy': combined.noise_policy.state_dict()}, path)
            payload = torch.load(path, weights_only=False)
        restored, _ = make_sum(history_length=payload['cfg'].noise_policy.history_length)
        restored.noise_policy.load_state_dict(payload['noise_policy'])
        restored.eval().reset(obs)
        torch.testing.assert_close(restored.predict_action(None, deterministic=True)['action'], expected)


if __name__ == '__main__':
    unittest.main()
