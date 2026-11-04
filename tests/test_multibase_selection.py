import io
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import dill
import gymnasium
import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from zprl.common.reactive_replay_buffer import ReactiveNStepReplayBuffer
from zprl.model.common.normalizer import SingleFieldLinearNormalizer
from zprl.policy.flow_match_unet_image_policy import FlowMatchUnetImagePolicy
from zprl.policy.residue_policy import ResiduePolicy, SumPolicy
from zprl.workspace import train_online_robomimic_workspace as workspace_module
from tests.test_reactive_n_step_replay import sample_at
from tests.test_reactive_policy import FakeVectorEnv


class TinyEncoder(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))
        self.width = width

    def output_shape(self):
        return (self.width,)

    def forward(self, obs):
        return obs['state'] * self.weight


class ConditionVelocity(torch.nn.Module):
    def forward(self, x, t, local_cond=None, global_cond=None):
        return global_cond[:, :1, None].expand_as(x)


def make_base(width=2, action_dim=2, n_obs_steps=2, n_action_steps=3):
    base = FlowMatchUnetImagePolicy(
        shape_meta={'action': {'shape': [action_dim]}}, obs_encoder=TinyEncoder(width),
        horizon=8, n_action_steps=n_action_steps, n_obs_steps=n_obs_steps,
        num_inference_steps=2, diffusion_step_embed_dim=8, down_dims=(8, 16), n_groups=2)
    base.model = ConditionVelocity()
    base.normalizer['state'] = SingleFieldLinearNormalizer.create_identity()
    base.normalizer['action'] = SingleFieldLinearNormalizer.create_manual(
        scale=torch.full((action_dim,), 2.), offset=torch.ones(action_dim),
        input_stats_dict={key: torch.ones(action_dim) for key in ('min', 'max', 'mean', 'std')})
    return base


def make_policy(**kwargs):
    config = dict(obs_dim=2, action_dim=2, hidden_dim=8, num_qs=3, num_subset=2,
                  num_base_candidates=3, num_res_candidates=1, include_base_candidate=True,
                  q_ent=False, res_scale=0.5, select_rollout=True, select_backup=True)
    config.update(kwargs)
    return ResiduePolicy(**config)


class DistanceQ(torch.nn.Module):
    def forward(self, obs, action):
        q = -(action[:, :1] - obs[:, :1]).square()
        return torch.stack([q, q + 1., -q - 100.])


class MultiBaseSelectionTest(unittest.TestCase):
    def test_decoder_k_one_and_k_many_keep_condition_horizon_and_normalization(self):
        base = make_base()
        obs = {'state': torch.arange(12.).reshape(2, 3, 2)}
        condition = base.encode_obs({'state': obs['state'][:, :2]})
        torch.manual_seed(4)
        original = base.predict_action(obs)
        torch.manual_seed(4)
        single = base.conditional_predict_k(condition, 1)
        for key in ('action', 'action_pred', 'naction', 'naction_pred'):
            torch.testing.assert_close(single[key][:, 0], original[key], rtol=0, atol=0)
        self.assertEqual(single['obs_emb'].shape, (2, 4))
        torch.manual_seed(5)
        many = base.conditional_predict_k(condition, 3)
        torch.manual_seed(5)
        flat = base.conditional_predict(condition.repeat_interleave(3, 0))
        self.assertEqual(many['naction_pred'].shape, (2, 3, 8, 2))
        self.assertEqual(many['naction'].shape, (2, 3, 3, 2))
        torch.testing.assert_close(many['naction'], many['naction_pred'][:, :, 1:4])
        torch.testing.assert_close(many['action_pred'], (many['naction_pred'] - 1.) / 2.)
        torch.testing.assert_close(many['naction_pred'].flatten(0, 1), flat['naction_pred'])
        self.assertFalse(torch.equal(many['naction_pred'][:, 0], many['naction_pred'][:, 1]))
        with self.assertRaises(AssertionError):
            base.conditional_predict_k(condition[:, :2], 3)
        base.obs_as_global_cond = False
        with self.assertRaises(ValueError):
            base.conditional_predict_k(condition, 3)

    def test_sum_encodes_once_and_keeps_full_decoder_embedding_separate_from_rl_state(self):
        base = make_base()
        policy = make_policy(obs_dim=3, action_dim=6)
        combined = SumPolicy(0.5, 2, 1, 2, 3, base, policy)
        obs = {'state': torch.arange(12.).reshape(2, 3, 2),
               'completed_stage_mask': torch.tensor([[[0.], [0.], [1.]], [[1.], [1.], [0.]]])}
        with patch.object(base.obs_encoder, 'forward', wraps=base.obs_encoder.forward) as encode, \
             patch.object(base, 'conditional_predict_k', wraps=base.conditional_predict_k) as decode:
            result, state = combined.encode_obs(obs)
        encode.assert_called_once()
        decode.assert_called_once()
        self.assertEqual(set(encode.call_args.args[0]), {'state'})
        torch.testing.assert_close(decode.call_args.args[0], obs['state'][:, :2].flatten(1))
        torch.testing.assert_close(state, torch.cat([obs['state'][:, 1], obs['completed_stage_mask'][:, -1]], -1))
        self.assertEqual(result['obs_emb'].shape, (2, 4))
        self.assertEqual(result['naction'].shape, (2, 3, 3, 2))
        combined.train()
        self.assertFalse(base.training)
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))

    def test_cross_base_pair_and_type_selection_shared_subset_and_ties(self):
        policy = make_policy()
        policy.q_targets = DistanceQ()
        obs = torch.tensor([[1.25, 0.], [2., 0.]])
        bank = torch.tensor([[[0., 10.], [1., 11.], [2., 12.]],
                             [[0., 20.], [1., 21.], [2., 22.]]])
        edits = torch.tensor([[[0.5, 0.1]], [[0.5, 0.2]], [[0.5, 0.3]],
                              [[0.5, 0.4]], [[0.5, 0.5]], [[0.5, 0.6]]])
        with patch.object(policy.actor, 'sample_candidates', return_value=edits) as actor, \
             patch.object(policy.q_targets, 'forward', wraps=policy.q_targets.forward) as q, \
             patch('torch.randperm', return_value=torch.tensor([0, 1, 2])) as subset:
            result = policy.select_base_res_pair(obs, bank)
        subset.assert_called_once()
        actor.assert_called_once()
        self.assertEqual(actor.call_args.args[1], 1)
        torch.testing.assert_close(actor.call_args.args[0][:, 2:], bank.flatten(0, 1))
        torch.testing.assert_close(actor.call_args.args[0][:, :2], obs.repeat_interleave(3, 0))
        self.assertEqual(q.call_args.args[1].shape, (12, 2))
        torch.testing.assert_close(result['base_index'], torch.tensor([1, 2]))
        torch.testing.assert_close(result['selected_is_res'], torch.tensor([True, False]))
        torch.testing.assert_close(result['base_naction'], bank[torch.arange(2), torch.tensor([1, 2])])
        torch.testing.assert_close(result['res_naction'], torch.tensor([[0.5, 0.2], [0., 0.]]))
        self.assertTrue(all(not value.requires_grad for value in result.values()))
        # Identical edits tie their own base; first original candidate wins.
        with patch.object(policy.actor, 'sample_candidates', return_value=torch.zeros_like(edits)), \
             patch('torch.randperm', return_value=torch.tensor([0, 1, 2])):
            result = policy.select_base_res_pair(torch.zeros(2, 2), torch.zeros_like(bank))
        torch.testing.assert_close(result['selected_index'], torch.zeros(2, dtype=torch.long))
        self.assertFalse(result['selected_is_res'].any())

    def test_bank_k_one_matches_stage_one_and_checkpoint_has_no_new_parameters(self):
        policy = make_policy(num_base_candidates=1, num_res_candidates=4)
        obs, base = torch.randn(2, 2), torch.randn(2, 2)
        torch.manual_seed(7)
        expected = policy._select_res_naction(obs, base)
        rng = torch.get_rng_state()
        torch.manual_seed(7)
        actual = policy.select_base_res_pair(obs, base[:, None])
        torch.testing.assert_close(actual['res_naction'], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual['q'], expected[1], rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        many = make_policy()
        many.load_state_dict(policy.state_dict(), strict=True)
        self.assertEqual(many.actor.net[0].in_features, 4)

    def test_rollout_switch_mask_and_eval_keep_executed_pair(self):
        base = make_base(width=2, action_dim=1, n_obs_steps=1, n_action_steps=2)
        bank = torch.tensor([[[[0.], [10.]], [[1.], [11.]], [[2.], [12.]]]]).repeat(2, 1, 1, 1)
        obs = torch.tensor([[1.25, 0.], [2., 0.]])
        for rollout in (False, True):
            for backup in (False, True):
                policy = make_policy(select_rollout=rollout, select_backup=backup)
                policy.q_targets = DistanceQ()
                combined = SumPolicy(0.5, 2, 0, 1, 2, base, policy)
                with patch.object(policy.actor, 'sample_candidates', return_value=torch.full((6, 1, 2), 0.5)), \
                     patch('torch.randperm', return_value=torch.tensor([0, 1, 2])), \
                     patch.object(combined, 'encode_obs', return_value=({'naction': bank}, obs)), \
                     patch.object(policy, 'select_base_res_pair', wraps=policy.select_base_res_pair) as selector:
                    torch.manual_seed(9)
                    result = combined.predict_train_action(bank, obs, torch.tensor([False, True]))
                    evaluated = combined.predict_action({})
                self.assertEqual(selector.call_count, 2 if rollout else 0)
                torch.testing.assert_close(result['base_naction_flat'][1], bank[1, 0].flatten())
                torch.testing.assert_close(result['res_naction_flat'][1], torch.zeros(2))
                torch.testing.assert_close(result['naction'].flatten(1), result['base_naction_flat'] + 0.5 * result['res_naction_flat'])
                if rollout:
                    torch.testing.assert_close(result['base_naction_flat'][0], bank[0, 1].flatten())
                    torch.testing.assert_close(result['res_naction_flat'][0], torch.full((2,), 0.5))
                else:
                    torch.testing.assert_close(result['base_naction_flat'], bank[:, 0].flatten(1))
                    mean = policy.actor.get_eval_action(torch.cat([obs, bank[:, 0].flatten(1)], -1))
                    expected = base.normalizer['action'].unnormalize((bank[:, 0].flatten(1) + 0.5 * mean).reshape(2, 2, 1))
                    torch.testing.assert_close(evaluated['action'], expected)
                    torch.manual_seed(9)
                    sample = policy.actor.get_action(torch.cat([obs, bank[:, 0].flatten(1)], -1))['sample']
                    torch.testing.assert_close(result['res_naction_flat'][0], sample[0])

    def test_backup_skips_terminal_banks_counts_types_and_preserves_target_formula(self):
        policy = make_policy(target_clip=[0., 1.])
        batch = ReplayBufferSamples(
            observations=torch.zeros(3, 2), next_observations=torch.ones(3, 2),
            actions=torch.zeros(3, 10), dones=torch.tensor([[0.], [1.], [0.]]),
            rewards=torch.tensor([[-0.2], [2.], [0.1]]), discounts=torch.tensor([[0.5], [0.2], [0.25]]))
        batch.actions[1, 4:] = float('nan')
        for types, ratio in [(torch.tensor([False, False]), 0.), (torch.tensor([True, True]), 1.),
                             (torch.tensor([True, False]), 0.5)]:
            # Explicit edited type must count even when its residual is exactly zero.
            selected = {'q': torch.tensor([[0.5], [2.]]), 'selected_is_res': types,
                        'res_naction': torch.zeros(2, 2)}
            with patch.object(policy, 'select_base_res_pair', return_value=selected) as selector:
                loss, info = policy.compute_critic_loss(batch)
            self.assertEqual(selector.call_args.args[1].shape, (2, 3, 2))
            self.assertTrue(torch.isfinite(selector.call_args.args[1]).all())
            self.assertEqual(info['backup_res_selection_ratio'], ratio)
            predicted = policy.qs(batch.observations, torch.zeros(3, 2)).squeeze(-1)
            torch.testing.assert_close(loss, ((predicted - torch.tensor([0.05, 1., 0.6])) ** 2).mean(-1).sum())
            loss.backward()
            self.assertTrue(all(p.grad is None for p in policy.actor.parameters()))
            policy.zero_grad(set_to_none=True)
        with patch.object(policy, 'select_base_res_pair', side_effect=AssertionError('empty selection')):
            loss, info = policy.compute_critic_loss(batch._replace(dones=torch.ones(3, 1)))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(info['backup_res_selection_ratio'], 0)

    def test_backup_ratio_uses_candidate_type_including_zero_edits(self):
        policy = make_policy()
        batch = ReplayBufferSamples(torch.zeros(2, 2), torch.zeros(2, 10), torch.ones(2, 2),
                                    torch.zeros(2, 1), torch.zeros(2, 1), torch.full((2, 1), 0.5))
        for winners, expected_ratio in [([0, 2], 0.), ([3, 5], 1.), ([0, 4], 0.5)]:
            scores = torch.zeros(2, 6)
            scores[torch.arange(2), winners] = 1.
            # Controlled heads distinguish candidate indices even for identical actions,
            # exercising edited-type accounting independently of residual magnitude.
            with patch.object(policy.actor, 'sample_candidates', return_value=torch.zeros(6, 1, 2)), \
                 patch.object(policy.q_targets, 'forward', return_value=scores.reshape(1, 12, 1).expand(3, -1, -1)):
                _, info = policy.compute_critic_loss(batch)
            self.assertEqual(info['backup_res_selection_ratio'], expected_ratio)

    def test_backup_off_and_actor_losses_use_saved_base_with_unchanged_gradients(self):
        many = make_policy(select_backup=False)
        one = make_policy(num_base_candidates=1, select_backup=False)
        one.load_state_dict(many.state_dict())
        batch = ReplayBufferSamples(torch.randn(2, 2), torch.randn(2, 10), torch.randn(2, 2),
                                    torch.zeros(2, 1), torch.randn(2, 1), torch.full((2, 1), 0.7))
        short = batch._replace(actions=batch.actions[:, :6])
        for method in ('compute_critic_loss', 'compute_actor_loss', 'compute_alpha_loss'):
            results = []
            for policy, data in [(many, batch), (one, short)]:
                policy.zero_grad(set_to_none=True)
                torch.manual_seed(13)
                result = getattr(policy, method)(data)
                loss = result[0] if isinstance(result, tuple) else result
                loss.backward()
                results.append((result, {n: p.grad for n, p in policy.named_parameters() if p.grad is not None}, torch.get_rng_state()))
            torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)

    def test_invalid_configs_and_unsupported_base(self):
        for override in [dict(num_base_candidates=0), dict(num_base_candidates=1.5),
                         dict(num_res_candidates=2), dict(num_res_candidates=0),
                         dict(include_base_candidate=False), dict(q_ent=True),
                         dict(actor_input='obs'), dict(critic_action_input='base_res')]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                make_policy(select_rollout=False, select_backup=False, **override)
        base = make_base(action_dim=1, n_action_steps=2)
        base.obs_as_global_cond = False
        with self.assertRaises(ValueError):
            SumPolicy(0.5, 2, 0, 1, 2, base, make_policy())


class MultiBaseReplayTest(unittest.TestCase):
    def buffer(self, k=3, n_steps=3, size=8, bootstrap=True):
        return ReactiveNStepReplayBuffer(size,
            gymnasium.spaces.Box(-np.inf, np.inf, (2 + k * 2,), dtype=np.float32),
            gymnasium.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32), device='cpu',
            n_envs=1, n_steps=n_steps, gamma=0.5, base_action_dim=2,
            num_base_candidates=k, handle_timeout_termination=bootstrap)

    def add(self, rb, t, done=False, timeout=False):
        k = rb.num_base_candidates
        state = [t, -t]
        base = [10 + t, 20 + t]
        bank = np.arange(k * 2, dtype=np.float32) + 100 * (t + 1)
        rb.add(np.array([state + base + [0.] * ((k - 1) * 2)], np.float32),
               np.array([[t + 1, -(t + 1), *bank]], np.float32),
               np.array([[t, -t]], np.float32), np.array([2. ** t], np.float32),
               np.array([done]), [{'TimeLimit.truncated': timeout}])

    def test_endpoint_bank_episode_boundary_timeout_and_wraparound(self):
        for k in (1, 3):
            for bootstrap in (False, True):
                rb = self.buffer(k=k, size=3, bootstrap=bootstrap)
                for t in range(4):
                    self.add(rb, t, done=t == 2, timeout=t == 2)
                # Start at t=1 (slot 1), stop at t=2 timeout; don't cross into t=3.
                batch = sample_at(rb, [1])
                self.assertEqual(rb.action_dim, 2)
                self.assertEqual(rb.bank_dim, k * 2)
                torch.testing.assert_close(batch.observations, torch.tensor([[1., -1.]]))
                torch.testing.assert_close(batch.next_observations, torch.tensor([[3., -3.]]))
                torch.testing.assert_close(batch.actions[:, :4], torch.tensor([[1., -1., 11., 21.]]))
                torch.testing.assert_close(batch.actions[:, 4:], (torch.arange(k * 2) + 300.).reshape(1, -1))
                self.assertEqual(batch.rewards.item(), 4.)
                self.assertEqual(batch.discounts.item(), 0.25)
                self.assertEqual(batch.dones.item(), float(not bootstrap))
                # Now exercise a complete n-step traversal crossing the ring boundary.
                rb = self.buffer(k=k, size=3)
                for t in range(4):
                    self.add(rb, t)
                batch = sample_at(rb, [1])
                torch.testing.assert_close(batch.next_observations, torch.tensor([[4., -4.]]))
                torch.testing.assert_close(batch.actions[:, 4:], (torch.arange(k * 2) + 400.).reshape(1, -1))
                self.assertEqual(batch.rewards.item(), 6.)
                self.assertEqual(batch.discounts.item(), 0.125)

    def test_padding_does_not_enter_any_loss_and_sampling_reuses_bank(self):
        rb = self.buffer(n_steps=1)
        self.add(rb, 0)
        original = sample_at(rb, [0])
        rb.observations[0, :, 4:] = float('nan')
        changed = sample_at(rb, [0])
        torch.testing.assert_close(original, changed)
        policy = make_policy()
        for method in ('compute_critic_loss', 'compute_actor_loss', 'compute_alpha_loss'):
            torch.manual_seed(17)
            a = getattr(policy, method)(original)
            torch.manual_seed(17)
            b = getattr(policy, method)(changed)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with patch.object(policy.actor, 'sample_candidates', wraps=policy.actor.sample_candidates) as actor:
            policy.compute_critic_loss(original)
            policy.compute_critic_loss(original)
        self.assertEqual(actor.call_count, 2)
        self.assertEqual(actor.call_args.args[0].shape, (3, 4))



class MultiBaseWorkspaceTest(unittest.TestCase):
    def test_storage_execution_timeout_cache_and_last_minibatch_learning_log(self):
        config_dir = pathlib.Path(__file__).resolve().parents[1] / 'zprl' / 'config'
        for bootstrap in ('never', 'truncated'):
            for rollout, backup in ((False, False), (False, True), (True, False), (True, True)):
                with self.subTest(bootstrap=bootstrap, rollout=rollout, backup=backup), tempfile.TemporaryDirectory() as tmp:
                    (pathlib.Path(tmp) / 'base.ckpt').touch()
                    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
                        cfg = compose(config_name='train_online_resrl_selection_workspace', overrides=[
                            f'online_task.base_ckpt={tmp}/base.ckpt', 'online_task.abs_action=false',
                            'online_task.shape_meta.action.shape=[1]', 'training.device=cpu',
                            'online_task.n_envs=3', 'training.training_freq=6',
                            'training.num_steps=6', 'training.learning_start=6',
                            'training.log_every=6', 'training.eval_every=6',
                            'training.checkpoint_every=6', 'training.utd=1',
                            'training.batch_size=2', 'training.buffer_size=12', 'training.n_step=2',
                            'res_policy.hidden_dim=8', 'res_policy.num_qs=2', 'res_policy.num_base_candidates=3',
                            f'res_policy.select_rollout={rollout}', f'res_policy.select_backup={backup}',
                            f'training.bootstrap_at_done={bootstrap}',
                        ])
                    base = make_base(width=1, action_dim=1, n_obs_steps=1, n_action_steps=4)
                    base.normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
                    payload = {'cfg': OmegaConf.create({'task_name': cfg.task_name, 'policy': {}}),
                               'state_dicts': {'ema_model': base.state_dict()}}
                    env, buffers, conditions, metrics = FakeVectorEnv(), [], [], []
                    instantiate = hydra.utils.instantiate
                    original_loss = ResiduePolicy.compute_critic_loss

                    def decode(condition, k):
                        conditions.append(condition.clone())
                        bank = (condition[:, None, None, :] * 10 + torch.arange(k)[None, :, None, None] * 1000
                                + torch.arange(4)[None, None, :, None])
                        return {'obs_emb': condition, 'naction': bank}

                    def instantiate_policy(config, **kwargs):
                        if '_target_' not in config:
                            return base
                        policy = instantiate(config, **kwargs)
                        # Prefer the largest base and its positive edit for every state.
                        policy.q_targets.forward = lambda obs, action: action[:, :1].unsqueeze(0).expand(2, -1, -1)
                        policy.actor.sample_candidates = lambda x, n: torch.full((x.shape[0], n, 4), 0.5)
                        return policy

                    def make_buffer(*args, **kwargs):
                        rb = ReactiveNStepReplayBuffer(*args, **kwargs)
                        buffers.append(rb)
                        return rb

                    def loss_with_ratios(policy, batch, dist=None):
                        loss, info = original_loss(policy, batch, dist)
                        if backup:
                            ratio = [1 / 3, 1., 0., 0., 0., 0.5][len(metrics)]
                            info['backup_res_selection_ratio'] = torch.tensor(ratio)
                            metrics.append(ratio)
                        return loss, info

                    with patch.object(workspace_module.torch, 'load', return_value=payload), \
                         patch.object(workspace_module.hydra.utils, 'instantiate', side_effect=instantiate_policy), \
                         patch.object(base, 'conditional_predict_k', side_effect=decode), \
                         patch.object(workspace_module.FileUtils, 'get_env_metadata_from_dataset', return_value={'env_kwargs': {}}), \
                         patch.object(workspace_module, 'AsyncVectorEnv', return_value=env), \
                         patch.object(workspace_module, 'ReactiveNStepReplayBuffer', side_effect=make_buffer), \
                         patch.object(ResiduePolicy, 'compute_critic_loss', new=loss_with_ratios), \
                         patch.object(workspace_module, 'wandb'):
                        workspace = workspace_module.TrainOnlineRobomimicWorkspace(cfg, tmp)
                        workspace.run()
                    rb = buffers[0]
                    self.assertEqual(rb.observations.shape, (4, 3, 15))
                    self.assertEqual(rb.actions.shape[-1], 4)
                    np.testing.assert_array_equal(rb.observations[:, :, 7:], 0.)
                    np.testing.assert_allclose(rb.observations[0, :, 3:7], np.array([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]))
                    np.testing.assert_array_equal(rb.actions[0], 0.)
                    selected = np.array([[100, 101, 102, 103], [200, 201, 202, 203], [40, 41, 42, 43]])
                    if rollout:
                        selected += 2000
                    np.testing.assert_allclose(rb.observations[1, :, 3:7], selected)
                    np.testing.assert_allclose(env.actions[1].reshape(3, 4), selected + cfg.training.res_scale * rb.actions[1])
                    batch = sample_at(rb, [0, 0, 0], [0, 1, 2])
                    next_states = [7, 20, 8] if bootstrap == 'truncated' else [10, 20, 30]
                    torch.testing.assert_close(batch.next_observations[:, 0], torch.tensor(next_states, dtype=torch.float32))
                    expected_bank = (np.array(next_states)[:, None, None] * 10 + np.arange(3)[None, :, None] * 1000
                                     + np.arange(4)[None, None, :]).reshape(3, 12)
                    np.testing.assert_allclose(batch.actions[:, 8:], expected_bank)
                    self.assertEqual(len(conditions), 5 if bootstrap == 'truncated' else 3)
                    if bootstrap == 'truncated':
                        torch.testing.assert_close(conditions[2], torch.tensor([[7.]]))
                        torch.testing.assert_close(conditions[4], torch.tensor([[8.]]))
                    log = json.loads((pathlib.Path(tmp) / 'logs.json.txt').read_text())
                    self.assertNotIn('info/backup_selection_count', log)
                    if backup:
                        self.assertEqual(log['info/backup_res_selection_ratio'], 0.5)
                    else:
                        self.assertNotIn('info/backup_res_selection_ratio', log)
                    self.assertFalse(base.training)
                    self.assertTrue(all(not p.requires_grad and p.grad is None for p in base.parameters()))

    def test_config_defaults_stage_one_overrides_and_checkpoint_roundtrip(self):
        config_dir = pathlib.Path(__file__).resolve().parents[1] / 'zprl' / 'config'
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name='train_online_resrl_selection_workspace', overrides=['res_policy.hidden_dim=8'])
            ordinary = compose(config_name='train_online_robomimic_workspace')
            first = compose(config_name='train_online_resrl_selection_workspace', overrides=[
                'res_policy.num_base_candidates=1', 'res_policy.num_res_candidates=8'])
        self.assertNotIn('num_base_candidates', ordinary.res_policy)
        self.assertEqual((cfg.res_policy.num_base_candidates, cfg.res_policy.num_res_candidates), (8, 1))
        self.assertEqual((first.res_policy.num_base_candidates, first.res_policy.num_res_candidates), (1, 8))
        policy = hydra.utils.instantiate(cfg.res_policy, obs_dim=2, action_dim=2)
        stream = io.BytesIO()
        torch.save({'cfg': cfg, 'res_policy': policy.state_dict()}, stream, pickle_module=dill)
        stream.seek(0)
        payload = torch.load(stream, pickle_module=dill)
        loaded = hydra.utils.instantiate(payload['cfg'].res_policy, obs_dim=2, action_dim=2)
        loaded.load_state_dict(payload['res_policy'], strict=True)
        bank, obs = torch.randn(2, 8, 2), torch.randn(2, 2)
        torch.manual_seed(19)
        expected = policy.select_base_res_pair(obs, bank)
        torch.manual_seed(19)
        torch.testing.assert_close(loaded.select_base_res_pair(obs, bank), expected)


if __name__ == '__main__':
    unittest.main()
