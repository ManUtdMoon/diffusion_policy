import unittest
from unittest.mock import patch

import gymnasium
import numpy as np
import torch
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from zprl.common.reactive_replay_buffer import ReactiveNStepReplayBuffer
from zprl.policy.residue_policy import ResiduePolicy


def make_buffer(n_steps, gamma=0.5, n_envs=1, buffer_size=16,
                handle_timeout_termination=True):
    obs_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
    action_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    return ReactiveNStepReplayBuffer(
        buffer_size,
        obs_space,
        action_space,
        device='cpu',
        n_envs=n_envs,
        handle_timeout_termination=handle_timeout_termination,
        n_steps=n_steps,
        gamma=gamma,
        base_action_dim=1,
    )


def add_step(buffer, obs, base, action, reward, next_obs, next_base,
             done=False, timeout=False):
    n_envs = buffer.n_envs

    def values(x):
        return np.asarray(x, dtype=np.float32).reshape(n_envs)

    obs = np.stack([values(obs), values(base)], axis=-1)
    next_obs = np.stack([values(next_obs), values(next_base)], axis=-1)
    action = values(action)[:, None]
    reward = values(reward)
    done = np.asarray(done, dtype=bool)
    if done.ndim == 0:
        done = np.full(n_envs, done, dtype=bool)
    timeout = np.asarray(timeout, dtype=bool)
    if timeout.ndim == 0:
        timeout = np.full(n_envs, timeout, dtype=bool)
    infos = [
        {'TimeLimit.truncated': bool(value)}
        for value in timeout
    ]
    buffer.add(obs, next_obs, action, reward, done, infos)


def sample_at(buffer, batch_inds, env_indices=None):
    batch_inds = np.asarray(batch_inds, dtype=np.int64)
    if env_indices is None:
        env_indices = np.zeros_like(batch_inds)
    with patch(
            'stable_baselines3.common.buffers.np.random.randint',
            return_value=np.asarray(env_indices, dtype=np.int64)):
        return buffer._get_samples(batch_inds)


class ReactiveNStepReplayBufferTest(unittest.TestCase):
    def test_n_one_preserves_existing_policy_batch_interface(self):
        buffer = make_buffer(n_steps=1)
        add_step(buffer, 1, 10, 3, 4, 2, 20)

        batch = sample_at(buffer, [0])

        torch.testing.assert_close(batch.observations, torch.tensor([[1.]]))
        torch.testing.assert_close(batch.next_observations, torch.tensor([[2.]]))
        torch.testing.assert_close(batch.actions, torch.tensor([[3., 10., 20.]]))
        torch.testing.assert_close(batch.rewards, torch.tensor([[4.]]))
        torch.testing.assert_close(batch.discounts, torch.tensor([[0.5]]))

    def test_n_two_and_three_accumulate_and_align_endpoint_base(self):
        for n_steps, reward, discount, next_obs, next_base in [
                (2, 2.0, 0.25, 2.0, 12.0),
                (3, 3.0, 0.125, 3.0, 13.0)]:
            with self.subTest(n_steps=n_steps):
                buffer = make_buffer(n_steps=n_steps)
                add_step(buffer, 0, 10, 100, 1, 1, 11)
                add_step(buffer, 1, 11, 101, 2, 2, 12)
                add_step(buffer, 2, 12, 102, 4, 3, 13)

                batch = sample_at(buffer, [0])

                torch.testing.assert_close(
                    batch.actions, torch.tensor([[100., 10., next_base]]))
                torch.testing.assert_close(
                    batch.next_observations, torch.tensor([[next_obs]]))
                torch.testing.assert_close(
                    batch.rewards, torch.tensor([[reward]]))
                torch.testing.assert_close(
                    batch.discounts, torch.tensor([[discount]]))

    def test_timeout_stops_return_and_controls_bootstrap_mask(self):
        for handle_timeout, expected_done in [(True, 0.), (False, 1.)]:
            with self.subTest(handle_timeout=handle_timeout):
                buffer = make_buffer(
                    n_steps=3,
                    handle_timeout_termination=handle_timeout)
                add_step(buffer, 0, 10, 100, 1, 1, 11)
                add_step(
                    buffer, 1, 11, 101, 2, 9, 99,
                    done=True, timeout=True)
                add_step(buffer, 100, 1000, 200, 100, 101, 1001)

                batch = sample_at(buffer, [0, 1, 2])

                torch.testing.assert_close(
                    batch.rewards, torch.tensor([[2.], [2.], [100.]]))
                torch.testing.assert_close(
                    batch.discounts, torch.tensor([[0.25], [0.5], [0.5]]))
                torch.testing.assert_close(
                    batch.next_observations, torch.tensor([[9.], [9.], [101.]]))
                torch.testing.assert_close(
                    batch.actions[:, 2:], torch.tensor([[99.], [99.], [1001.]]))
                torch.testing.assert_close(
                    batch.dones.flatten(),
                    torch.tensor([expected_done, expected_done, 0.]))

    def test_wraparound_keeps_endpoint_alignment(self):
        buffer = make_buffer(n_steps=3, buffer_size=3)
        for t, reward in enumerate([1., 2., 4., 8.]):
            add_step(buffer, t, 10 + t, 100 + t, reward,
                     t + 1, 11 + t)

        batch = sample_at(buffer, [1])

        torch.testing.assert_close(batch.rewards, torch.tensor([[6.]]))
        torch.testing.assert_close(batch.discounts, torch.tensor([[0.125]]))
        torch.testing.assert_close(batch.next_observations, torch.tensor([[4.]]))
        torch.testing.assert_close(batch.actions, torch.tensor([[101., 11., 14.]]))

    def test_multiple_environments_do_not_mix(self):
        buffer = make_buffer(n_steps=2, n_envs=2, buffer_size=8)
        add_step(
            buffer,
            obs=[0, 100], base=[10, 1000], action=[1, 10],
            reward=[1, 100], next_obs=[1, 101], next_base=[11, 1001])
        add_step(
            buffer,
            obs=[1, 101], base=[11, 1001], action=[2, 20],
            reward=[2, 200], next_obs=[2, 102], next_base=[12, 1002])

        batch = sample_at(buffer, [0, 0], env_indices=[0, 1])

        torch.testing.assert_close(batch.rewards, torch.tensor([[2.], [200.]]))
        torch.testing.assert_close(
            batch.actions, torch.tensor([[1., 10., 12.], [10., 1000., 1002.]]))


class ConstantQ(torch.nn.Module):
    def __init__(self, value, num_qs):
        super().__init__()
        self.value = value
        self.num_qs = num_qs

    def forward(self, obs, action):
        return torch.full(
            (self.num_qs, obs.shape[0], 1),
            self.value,
            device=obs.device,
            dtype=obs.dtype,
        )


class ResiduePolicyDiscountTest(unittest.TestCase):
    def test_critic_uses_per_sample_discounts_and_one_step_fallback(self):
        policy = ResiduePolicy(
            obs_dim=1,
            action_dim=1,
            hidden_dim=8,
            gamma=0.9,
            auto_alpha=False,
            q_ent=False,
            num_qs=2,
            num_subset=2,
        )
        policy.qs = ConstantQ(0., 2)
        policy.q_targets = ConstantQ(4., 2)
        batch = ReplayBufferSamples(
            observations=torch.zeros(2, 1),
            actions=torch.zeros(2, 3),
            next_observations=torch.zeros(2, 1),
            dones=torch.tensor([[0.], [1.]]),
            rewards=torch.tensor([[1.], [2.]]),
            discounts=torch.tensor([[0.25], [0.5]]),
        )

        def sample_action(actor_input):
            bs = actor_input.shape[0]
            zeros = torch.zeros(bs, 1)
            return zeros, zeros, zeros

        with patch.object(
                policy, '_sample_naction_log_prob',
                side_effect=sample_action):
            loss, info = policy.compute_critic_loss(batch)
            fallback_loss, fallback_info = policy.compute_critic_loss(
                batch._replace(discounts=None))

        torch.testing.assert_close(loss, torch.tensor(8.))
        self.assertAlmostEqual(info['q_target'], 2.)
        torch.testing.assert_close(fallback_loss, torch.tensor(25.16))
        self.assertAlmostEqual(fallback_info['q_target'], 3.3)


if __name__ == '__main__':
    unittest.main()
