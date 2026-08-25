import pathlib
import unittest
from types import SimpleNamespace

import gym
import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from zprl.env.adroit.adroit import AdroitEarlyStopWrapper
from zprl.gym_util.multistep_wrapper import MultiStepWrapper
from zprl.model.online import (
    BatchedLayerNorm, BatchedSoftQNet, BatchedUnboundedQNet)
from zprl.policy.latent_policy import ResiduePolicy as LatentResiduePolicy
from zprl.policy.noise_policy import DSRLQNet, NoisePolicy
from zprl.policy.residue_policy import ResiduePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)


class _FixedQ(nn.Module):
    def __init__(self, num_qs, value):
        super().__init__()
        self.num_qs = num_qs
        self.value = value

    def forward(self, obs, action):
        return torch.full(
            (self.num_qs, obs.shape[0], 1), self.value,
            device=obs.device, dtype=obs.dtype)


class _RepeatScaledRewardEnv(gym.Env):
    def __init__(self, low_level_rewards, scale, stop_after_first=False):
        self.low_level_rewards = list(low_level_rewards)
        self.scale = scale
        self.stop_after_first = stop_after_first
        self.low_level_steps = 0
        self.wrapper_steps = 0
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })

    def reset(self):
        self.low_level_steps = 0
        self.wrapper_steps = 0
        return {'state': np.zeros(1, dtype=np.float32)}

    def step(self, action):
        rewards = self.low_level_rewards[
            self.low_level_steps:self.low_level_steps + 2]
        self.low_level_steps += len(rewards)
        self.wrapper_steps += 1
        info = {'door_pos': -0.2 if self.stop_after_first else 0.0}
        obs = {'state': np.array([self.low_level_steps], dtype=np.float32)}
        return obs, sum(rewards) * self.scale, False, info


class AdroitMetaworldCriticRewardTest(unittest.TestCase):
    def test_unbounded_ensemble_has_per_q_layer_norm_and_raw_output(self):
        for q_cls in (BatchedUnboundedQNet, DSRLQNet):
            q = q_cls(obs_dim=3, action_dim=2, num_qs=3, hidden_dim=8)
            norms = [m for m in q.modules() if isinstance(m, BatchedLayerNorm)]
            self.assertEqual(len(norms), 3)
            for norm in norms:
                self.assertEqual(norm.weight.shape, (3, 1, 8))
                self.assertEqual(norm.bias.shape, (3, 1, 8))
            with torch.no_grad():
                norms[0].weight[1].fill_(2.0)
                norms[0].bias[2].fill_(1.0)
                q.net[-1].weight.zero_()
                q.net[-1].bias[0].fill_(-2.0)
                q.net[-1].bias[1].fill_(2.0)
            self.assertFalse(torch.equal(norms[0].weight[0], norms[0].weight[1]))
            output = q(torch.randn(5, 3), torch.randn(5, 2))
            self.assertEqual(output.shape, (3, 5, 1))
            self.assertLess(output.min().item(), 0.0)
            self.assertGreater(output.max().item(), 1.0)

    def test_robomimic_default_critic_remains_bounded(self):
        for policy in (
                ResiduePolicy(obs_dim=3, action_dim=2, hidden_dim=8),
                LatentResiduePolicy(
                    obs_dim=3, z_dim=2, action_dim=2, hidden_dim=8)):
            self.assertIsInstance(policy.qs, BatchedSoftQNet)
            output = policy.qs(
                torch.randn(5, policy.qs.net[0].weight.shape[1] - 2),
                torch.randn(5, 2))
            self.assertTrue(torch.all(output >= 0.0))
            self.assertTrue(torch.all(output <= 1.0))

    def _assert_policy_losses(self, policy, batch):
        optimizers = policy.get_optimizer(1e-4, 3e-4)
        critic_loss, critic_info = policy.compute_critic_loss(batch)
        self.assertTrue(torch.isfinite(critic_loss))
        optimizers['q_optimizer'].zero_grad()
        critic_loss.backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in policy.qs.parameters()))
        optimizers['q_optimizer'].step()

        actor_loss, _ = policy.compute_actor_loss(batch)
        self.assertTrue(torch.isfinite(actor_loss))
        optimizers['actor_optimizer'].zero_grad()
        actor_loss.backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in policy.actor.parameters()))
        optimizers['actor_optimizer'].step()
        for key in (
                'rewards_min', 'rewards', 'rewards_max',
                'q_target_min', 'q_target', 'q_target_max',
                'q_predicted_min', 'q_predicted', 'q_predicted_max'):
            self.assertTrue(np.isfinite(critic_info[key]))

    def test_three_policy_losses_and_gradients_are_finite(self):
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            res = ResiduePolicy(
                obs_dim=3, action_dim=2, hidden_dim=8,
                auto_alpha=False, unbounded_q=True)
            self._assert_policy_losses(res, SimpleNamespace(
                observations=torch.randn(4, 3),
                next_observations=torch.randn(4, 3),
                actions=torch.randn(4, 6),
                rewards=torch.tensor([[-2.0], [0.0], [1.0], [3.0]]),
                dones=torch.zeros(4, 1)))

            latent = LatentResiduePolicy(
                obs_dim=3, z_dim=2, action_dim=2, hidden_dim=8,
                auto_alpha=False, unbounded_q=True)
            self._assert_policy_losses(latent, SimpleNamespace(
                observations=torch.randn(4, 9),
                next_observations=torch.randn(4, 9),
                actions=torch.randn(4, 2),
                rewards=torch.tensor([[-2.0], [0.0], [1.0], [3.0]]),
                dones=torch.zeros(4, 1)))

            noise = NoisePolicy(
                obs_dim=3, action_dim=2, hidden_dim=8,
                auto_alpha=False, q_entropy=False)
            self._assert_policy_losses(noise, SimpleNamespace(
                observations=torch.randn(4, 3),
                next_observations=torch.randn(4, 3),
                actions=torch.randn(4, 2),
                rewards=torch.tensor([[-2.0], [0.0], [1.0], [3.0]]),
                dones=torch.zeros(4, 1)))

    def test_metaworld_scale_target_is_not_saturated(self):
        policy = ResiduePolicy(
            obs_dim=3, action_dim=2, hidden_dim=8, gamma=0.99,
            auto_alpha=False, unbounded_q=True)
        policy.q_targets = _FixedQ(2, 3.0)
        policy.qs = _FixedQ(2, 0.0)
        policy._sample_naction_log_prob = lambda obs: (
            torch.zeros(obs.shape[0], 2), torch.zeros(obs.shape[0], 1))
        _, info = policy.compute_critic_loss(SimpleNamespace(
            observations=torch.zeros(2, 3),
            next_observations=torch.zeros(2, 3),
            actions=torch.zeros(2, 6),
            rewards=torch.tensor([[-4.0], [2.0]]),
            dones=torch.zeros(2, 1)))
        self.assertAlmostEqual(info['q_target_min'], -1.03, places=5)
        self.assertAlmostEqual(info['q_target_max'], 4.97, places=5)

    def test_adroit_reward_contract_and_early_stop(self):
        env = MultiStepWrapper(
            _RepeatScaledRewardEnv([10.0, 20.0, 30.0, 40.0], 1 / 20),
            n_obs_steps=1, n_action_steps=2,
            reward_agg_method='discounted_sum', gamma=0.99,
            reward_offset=-1.0)
        env.reset()
        _, reward, done, info = env.step(
            np.zeros((2, 1), dtype=np.float32))
        self.assertFalse(done)
        self.assertEqual(env.env.low_level_steps, 4)
        self.assertAlmostEqual(info['raw_reward'], 3.5)
        self.assertAlmostEqual(reward, 0.5 + 0.99 * 2.5)

        early_env = MultiStepWrapper(
            AdroitEarlyStopWrapper(_RepeatScaledRewardEnv(
                [10.0, 20.0, 30.0, 40.0], 1 / 20,
                stop_after_first=True)),
            n_obs_steps=1, n_action_steps=2,
            reward_agg_method='discounted_sum', gamma=0.99)
        early_env.reset()
        _, early_reward, early_done, _ = early_env.step(
            np.zeros((2, 1), dtype=np.float32))
        self.assertTrue(early_done)
        self.assertEqual(early_env.env.env.low_level_steps, 2)
        self.assertAlmostEqual(early_reward, 1.5)

    def test_hydra_opt_in_is_limited_to_mw_adroit_configs(self):
        for name in (
                'train_online_workspace', 'train_online_vib_workspace',
                'train_online_noise_workspace'):
            GlobalHydra.instance().clear()
            with initialize(version_base=None, config_path='../zprl/config'):
                cfg = compose(config_name=name)
            self.assertEqual(cfg.single_gamma, 0.99)
            policy_cfg = cfg.noise_policy if name.endswith('noise_workspace') \
                else cfg.res_policy
            self.assertAlmostEqual(
                policy_cfg.gamma,
                round(cfg.single_gamma ** cfg.n_action_steps, 2))
            if name.endswith('noise_workspace'):
                self.assertEqual(
                    policy_cfg._target_, 'zprl.policy.noise_policy.NoisePolicy')
                self.assertEqual(cfg.training.reward_offset, 0.0)
            else:
                self.assertTrue(policy_cfg.unbounded_q)

        root = pathlib.Path(__file__).parents[1]
        for name in (
                'train_online_robomimic_workspace.yaml',
                'train_online_vib_robomimic_workspace.yaml'):
            source = (root / 'zprl/config' / name).read_text()
            self.assertNotIn('unbounded_q', source)

        for name in (
                'train_online_workspace.py',
                'train_online_vib_workspace.py',
                'train_online_noise_workspace.py'):
            source = (root / 'zprl/workspace' / name).read_text()
            for suffix in ('min', 'mean', 'max'):
                self.assertIn(
                    f'info/replay_discounted_chunk_reward_{suffix}', source)


if __name__ == '__main__':
    unittest.main()
