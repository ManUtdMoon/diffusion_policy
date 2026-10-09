import copy
import unittest
from unittest.mock import patch

import gym
import gymnasium
import numpy as np
import torch

from zprl.common.reactive_history import HistoryWindow
from zprl.common.reactive_history_replay_buffer import ReactiveHistoryReplayBuffer
from zprl.gym_util.multistep_wrapper import MultiStepWrapper
from zprl.policy.reactive_history_policy import HistoryResiduePolicy, HistorySumPolicy


class IdentityNormalizer:
    def unnormalize(self, action):
        return action


class AffineNormalizer:
    def unnormalize(self, action):
        return action * 3 + 5


class FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.normalizer = {'action': IdentityNormalizer()}
        self.predict_calls = []
        self.encode_calls = []

    def encode_obs(self, obs):
        self.encode_calls.append(obs['state'][:, -1, 0].clone())
        return obs['state'][:, -1]

    def predict_action(self, obs):
        self.predict_calls.append(obs['state'][:, -1, 0].clone())
        value = obs['state'][:, -1, 0]
        plan = value[:, None, None] * 10 + torch.arange(4, device=value.device)[None, :, None]
        return {'obs_emb': obs['state'][:, -1], 'naction': plan}


def make_obs(values, masks=None):
    values = np.asarray(values, dtype=np.float32)
    if masks is None:
        masks = np.zeros_like(values)
    masks = np.asarray(masks, dtype=np.float32)
    return {'state': values[:, None, None],
            'completed_stage_mask': masks[:, None, None]}


def make_policy():
    residual = HistoryResiduePolicy(
        obs_dim=2, action_dim=2, history_length=4, hidden_dim=8,
        num_qs=2)
    base = FakeBase()
    return HistorySumPolicy(0.5, 1, 1, 1, 4, base, residual, 2), base


class HistoryWindowTest(unittest.TestCase):
    def test_complete_append_and_partial_reset(self):
        window = HistoryWindow(3, 2)
        window.reset(torch.tensor([[1.], [10.]]))
        window.complete(torch.tensor([[2., 3.], [20., 30.]]), torch.tensor([2, 1]))
        window.append(torch.tensor([[4.], [40.]]))
        obs, action, valid, steps = window.get()
        torch.testing.assert_close(obs[:, -2:, 0], torch.tensor([[1., 4.], [10., 40.]]))
        torch.testing.assert_close(action[:, -2], torch.tensor([[2., 3.], [20., 30.]]))
        torch.testing.assert_close(action[:, -1], torch.zeros(2, 2))
        torch.testing.assert_close(steps[:, -2:], torch.tensor([[2, 0], [1, 0]]))
        window.reset(torch.tensor([[4.], [50.]]), torch.tensor([False, True]))
        new_obs, _, new_valid, _ = window.get()
        torch.testing.assert_close(new_obs[0], obs[0])
        torch.testing.assert_close(new_valid[1], torch.tensor([False, False, True]))
        torch.testing.assert_close(new_obs[1, -1, 0], torch.tensor(50.))
        self.assertTrue(valid[:, -1].all())


class HistorySumRolloutTest(unittest.TestCase):
    def test_predict_feedback_timeout_and_partial_reset(self):
        policy, base = make_policy()
        with self.assertRaisesRegex(RuntimeError, 'reset'):
            policy.predict_action(None)
        policy.reset(make_obs([1, 2], [0, 1]))
        self.assertEqual(len(base.predict_calls), 1)
        initial_window = policy.history.get()
        calls = []

        def residual(obs, a_exec, obs_valid, a_base, argmax=False):
            calls.append((argmax, obs.clone(), a_exec.clone(), obs_valid.clone()))
            return torch.ones(obs.shape[0], 2)

        with patch.object(policy.res_policy, 'predict_res_naction', side_effect=residual):
            output = policy.predict_action({'unused': object()}, deterministic=False,
                                           res_mask=torch.tensor([True, False]))
            self.assertFalse(calls[0][0])
            torch.testing.assert_close(output['naction'][:, :, 0],
                                       torch.tensor([[10., 11.], [20.5, 21.5]]))
            self.assertEqual(len(base.predict_calls), 1)
            self.assertEqual(len(base.encode_calls), 0)
            for before, after in zip(initial_window, policy.history.get()):
                torch.testing.assert_close(before, after)
            with self.assertRaisesRegex(RuntimeError, 'on_env_step'):
                policy.predict_action(None)
            infos = [
                {'executed_action_steps': 2},
                {'executed_action_steps': 1, 'TimeLimit.truncated': True,
                 'terminal_observation': {key: value[0] for key, value in
                                          make_obs([99], [1]).items()}},
            ]
            transition = policy.on_env_step(make_obs([3, 4], [0, 0]),
                                            [False, True], infos)
            torch.testing.assert_close(transition['obs'], torch.tensor([[1., 0.], [2., 1.]]))
            torch.testing.assert_close(transition['action'][0], torch.zeros(2))
            torch.testing.assert_close(transition['a_exec'][1], torch.tensor([20.5, 21.5]))
            torch.testing.assert_close(transition['exec_steps'], torch.tensor([2, 1]))
            torch.testing.assert_close(transition['next_obs'], torch.tensor([[3., 0.], [99., 1.]]))
            torch.testing.assert_close(transition['next_a_base'], torch.tensor([[12., 13.], [22., 23.]]))
            obs, action, valid, steps = policy.history.get()
            torch.testing.assert_close(obs[0, -2:], torch.tensor([[1., 0.], [3., 0.]]))
            torch.testing.assert_close(action[0, -2], transition['a_exec'][0])
            torch.testing.assert_close(valid[1], torch.tensor([False, False, False, True]))
            torch.testing.assert_close(obs[1, -1], torch.tensor([4., 0.]))
            with self.assertRaisesRegex(RuntimeError, 'predict_action'):
                policy.on_env_step(make_obs([3, 4]), [False, False], infos)
            policy.predict_action({'unused': object()}, deterministic=True)
            self.assertTrue(calls[1][0])
            torch.testing.assert_close(policy.pending['a_base'],
                                       torch.tensor([[12., 13.], [40., 41.]]))
            self.assertEqual(len(base.predict_calls), 2)
            self.assertEqual(len(base.encode_calls), 2)
            self.assertEqual(steps[0, -2].item(), 2)

    def test_terminal_boundary_uses_terminal_plan_and_reset_uses_new_plan(self):
        policy, base = make_policy()
        policy.reset(make_obs([1]))
        with patch.object(policy.res_policy, 'predict_res_naction',
                          return_value=torch.zeros(1, 2)):
            policy.predict_action(None)
            policy.on_env_step(make_obs([2]), [False], [{'executed_action_steps': 2}])
            policy.predict_action(None)
            terminal = {key: value[0] for key, value in make_obs([77]).items()}
            transition = policy.on_env_step(
                make_obs([8]), [True], [{'executed_action_steps': 1,
                                        'TimeLimit.truncated': True,
                                        'terminal_observation': terminal}])
        torch.testing.assert_close(transition['next_obs'], torch.tensor([[77., 0.]]))
        torch.testing.assert_close(transition['next_a_base'], torch.tensor([[770., 771.]]))
        torch.testing.assert_close(policy.base_cache[0, :2, 0], torch.tensor([80., 81.]))
        torch.testing.assert_close(policy.history.get()[2][0],
                                   torch.tensor([False, False, False, True]))
        self.assertEqual(len(base.predict_calls), 3)

    def test_replay_endpoint_matches_online_window_and_immediate_write(self):
        policy, base = make_policy()
        base.normalizer['action'] = AffineNormalizer()
        policy.reset(make_obs([1]))
        with patch.object(policy.res_policy, 'predict_res_naction',
                          return_value=torch.ones(1, 2)):
            output = policy.predict_action(None, deterministic=False)
            transition = policy.on_env_step(make_obs([2]), [False],
                                            [{'executed_action_steps': 2}])
        torch.testing.assert_close(output['naction'].flatten(), torch.tensor([10.5, 11.5]))
        torch.testing.assert_close(output['action'].flatten(), torch.tensor([36.5, 39.5]))
        torch.testing.assert_close(transition['a_base'], torch.tensor([[10., 11.]]))
        torch.testing.assert_close(transition['action'], torch.ones(1, 2))
        torch.testing.assert_close(transition['a_exec'], torch.tensor([[10.5, 11.5]]))
        torch.testing.assert_close(transition['next_a_base'], torch.tensor([[12., 13.]]))
        obs_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        action_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        buffer = ReactiveHistoryReplayBuffer(
            8, obs_space, action_space, device='cpu', n_steps=1, gamma=0.5,
            base_action_dim=2, action_step_dim=1, history_length=4)
        buffer.add(
            transition['obs'].numpy(), transition['next_obs'].numpy(),
            transition['action'].numpy(), np.array([1.], dtype=np.float32),
            np.array([False]), [{'executed_action_steps': 2}],
            a_base=transition['a_base'].numpy(),
            next_a_base=transition['next_a_base'].numpy(),
            a_exec=transition['a_exec'].numpy(),
            exec_steps=transition['exec_steps'].numpy(), episode_id=np.array([0]))
        with patch('stable_baselines3.common.buffers.np.random.randint',
                   return_value=np.array([0])):
            sample = buffer._get_samples(np.array([0]))
        online = policy.history.get()
        torch.testing.assert_close(sample.next_obs, online[0])
        torch.testing.assert_close(sample.next_a_exec, online[1])
        torch.testing.assert_close(sample.next_obs_valid, online[2])
        torch.testing.assert_close(sample.next_exec_steps, online[3])
        torch.testing.assert_close(sample.a_base, transition['a_base'])
        torch.testing.assert_close(sample.action, transition['action'])
        torch.testing.assert_close(sample.next_a_base, transition['next_a_base'])
        policy.res_policy.compute_critic_loss(sample)
        for before, after in zip(online, policy.history.get()):
            torch.testing.assert_close(before, after)

    def test_train_eval_copies_do_not_share_models_or_session_state(self):
        train, _ = make_policy()
        evaluation = copy.deepcopy(train)
        self.assertNotEqual(train.res_policy.actor.net[0].weight.data_ptr(),
                            evaluation.res_policy.actor.net[0].weight.data_ptr())
        self.assertNotEqual(train.base_policy.weight.data_ptr(), evaluation.base_policy.weight.data_ptr())
        self.assertTrue(all(not p.requires_grad for p in train.base_policy.parameters()))
        self.assertTrue(all(not p.requires_grad for p in evaluation.base_policy.parameters()))
        train.reset(make_obs([1]))
        evaluation.reset(make_obs([9]))
        with patch.object(train.res_policy, 'predict_res_naction',
                          return_value=torch.zeros(1, 2)):
            train.predict_action(None, deterministic=False)
        self.assertIsNone(evaluation.pending)
        torch.testing.assert_close(evaluation.history.get()[0][0, -1], torch.tensor([9., 0.]))
        torch.testing.assert_close(evaluation.base_cache[0, :2, 0], torch.tensor([90., 91.]))
        with torch.no_grad():
            train.res_policy.actor.mean.weight.add_(1)
        self.assertFalse(torch.equal(train.res_policy.actor.mean.weight,
                                     evaluation.res_policy.actor.mean.weight))
        evaluation.res_policy.load_state_dict(train.res_policy.state_dict())
        torch.testing.assert_close(train.res_policy.actor.mean.weight,
                                   evaluation.res_policy.actor.mean.weight)
        evaluation.eval()
        self.assertFalse(evaluation.base_policy.training)
        self.assertFalse(evaluation.res_policy.actor.training)
        self.assertIsNotNone(train.pending)
        evaluation.reset(make_obs([7]))
        torch.testing.assert_close(train.history.get()[0][0, -1], torch.tensor([1., 0.]))
        torch.testing.assert_close(train.base_cache[0, :2, 0], torch.tensor([10., 11.]))

    def test_true_terminal_resets_without_terminal_observation(self):
        policy, _ = make_policy()
        policy.reset(make_obs([1]))
        with patch.object(policy.res_policy, 'predict_res_naction',
                          return_value=torch.zeros(1, 2)):
            policy.predict_action(None)
            transition = policy.on_env_step(make_obs([5]), [True],
                                            [{'executed_action_steps': 1}])
        torch.testing.assert_close(transition['exec_steps'], torch.tensor([1]))
        torch.testing.assert_close(policy.history.get()[2][0],
                                   torch.tensor([False, False, False, True]))
        torch.testing.assert_close(policy.base_cache[0, :2, 0], torch.tensor([50., 51.]))


class FakePrimitiveEnv(gym.Env):
    def __init__(self, done_at=None):
        super().__init__()
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)})
        self.done_at = done_at
        self.steps = 0

    def reset(self):
        self.steps = 0
        return {'state': np.array([0.], dtype=np.float32)}

    def step(self, action):
        self.steps += 1
        obs = {'state': np.array([self.steps], dtype=np.float32)}
        return obs, 1., self.steps == self.done_at, {'task_reward': 0.}


class MultiStepFeedbackTest(unittest.TestCase):
    def test_executed_steps_reports_timeout_terminal_and_full_chunk(self):
        for done_at, limit, expected in ((None, 2, 2), (1, None, 1), (None, None, 4)):
            with self.subTest(done_at=done_at, limit=limit):
                env = MultiStepWrapper(FakePrimitiveEnv(done_at), 1, 4,
                                       max_episode_steps=limit)
                env.reset()
                action = np.arange(4, dtype=np.float32)[:, None]
                _, _, done, info = env.step(action)
                self.assertEqual(info['executed_action_steps'], expected)
                self.assertEqual(done, expected < 4)
                np.testing.assert_array_equal(action[:, 0], np.arange(4))


if __name__ == '__main__':
    unittest.main()
