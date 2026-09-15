import unittest
from types import SimpleNamespace

import gym
from gym import spaces
import numpy as np

from zprl.env.robomimic.robomimic_square_subtask_wrapper import (
    SquareSubtaskWrapper,
    get_subtask_dim,
)
from zprl.gym_util.multistep_wrapper import MultiStepWrapper


class FakeSquareTask:
    single_object_mode = 2
    nut_to_id = {'square': 0, 'round': 1}
    nut_id = 0

    def __init__(self):
        self.rewards = (0.0, 0.0, 0.0, 0.0)

    def staged_rewards(self):
        return self.rewards


class FakeSquareEnv(gym.Env):
    def __init__(self, transitions):
        self.env = FakeSquareTask()
        self.transitions = list(transitions)
        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self):
        return {'state': np.zeros(1, dtype=np.float32)}

    def step(self, action):
        transition = self.transitions.pop(0)
        staged_rewards, task_reward = transition[:2]
        done, info = transition[2:] if len(transition) > 2 else (False, {})
        self.env.rewards = staged_rewards
        obs = {'state': np.zeros(1, dtype=np.float32)}
        return obs, task_reward, done, info


def make_config(stages, weights, reward_mode='semi_sparse', enabled=True):
    return SimpleNamespace(
        enabled=enabled,
        stages=stages,
        reward_mode=reward_mode,
        reward_scale=1.0,
        stage_weights=weights,
        hover_threshold=0.65,
        obs_key='completed_stage_mask',
    )


class SquareSubtaskWrapperTest(unittest.TestCase):
    def test_grasp_is_rewarded_once_and_success_uses_task_reward(self):
        env = FakeSquareEnv([
            ((0.0, 0.35, 0.0, 0.0), 0.0),
            ((0.0, 0.35, 0.0, 0.0), 0.0),
            ((0.0, 0.0, 0.0, 0.0), 1.0),
            ((0.0, 0.0, 0.0, 0.0), 0.0),
        ])
        wrapper = SquareSubtaskWrapper(env, make_config(['grasp'], [1.0]))

        obs = wrapper.reset()
        np.testing.assert_array_equal(obs['completed_stage_mask'], [0.0])

        obs, reward, _, info = wrapper.step(np.zeros(1))
        self.assertEqual(reward, 1.0)
        np.testing.assert_array_equal(obs['completed_stage_mask'], [1.0])
        np.testing.assert_array_equal(info['stage_completion_delta'], [1.0])

        _, reward, _, info = wrapper.step(np.zeros(1))
        self.assertEqual(reward, 0.0)
        np.testing.assert_array_equal(info['stage_completion_delta'], [0.0])

        obs, reward, _, info = wrapper.step(np.zeros(1))
        self.assertEqual(reward, 1.0)
        self.assertEqual(info['stage_reward'], 0.0)
        self.assertEqual(info['task_reward'], 1.0)
        np.testing.assert_array_equal(obs['completed_stage_mask'], [1.0])

    def test_checklist_allows_independent_stage_completion(self):
        env = FakeSquareEnv([
            ((0.0, 0.0, 0.0, 0.7), 0.0),
            ((0.0, 0.35, 0.0, 0.0), 0.0),
            ((0.0, 0.35, 0.0, 0.65), 0.0),
        ])
        wrapper = SquareSubtaskWrapper(
            env, make_config(['grasp', 'hover'], [1.0, 1.0]))
        wrapper.reset()

        obs, reward, _, info = wrapper.step(np.zeros(1))
        self.assertEqual(reward, 1.0)
        np.testing.assert_array_equal(obs['completed_stage_mask'], [0.0, 1.0])
        np.testing.assert_array_equal(info['stage_completion_delta'], [0.0, 1.0])

        obs, reward, _, info = wrapper.step(np.zeros(1))
        self.assertEqual(reward, 1.0)
        np.testing.assert_array_equal(obs['completed_stage_mask'], [1.0, 1.0])
        np.testing.assert_array_equal(info['stage_completion_delta'], [1.0, 0.0])

        obs, reward, _, info = wrapper.step(np.zeros(1))
        self.assertEqual(reward, 0.0)
        np.testing.assert_array_equal(obs['completed_stage_mask'], [1.0, 1.0])
        np.testing.assert_array_equal(info['stage_completion_delta'], [0.0, 0.0])

    def test_sparse_mode_updates_checklist_without_stage_reward(self):
        env = FakeSquareEnv([((0.0, 0.35, 0.0, 0.0), 0.0)])
        wrapper = SquareSubtaskWrapper(
            env, make_config(['grasp'], [1.0], reward_mode='sparse'))
        wrapper.reset()

        obs, reward, _, info = wrapper.step(np.zeros(1))
        self.assertEqual(reward, 0.0)
        self.assertEqual(info['stage_reward'], 0.0)
        np.testing.assert_array_equal(obs['completed_stage_mask'], [1.0])

    def test_disabled_preserves_mdp_and_tracks_stages(self):
        env = FakeSquareEnv([((0.0, 0.35, 0.0, 0.0), 0.0)])
        config = make_config(['grasp'], [1.0], enabled=False)
        wrapper = SquareSubtaskWrapper(env, config)

        self.assertEqual(get_subtask_dim(config), 0)
        obs = wrapper.reset()
        self.assertNotIn(config.obs_key, obs)
        self.assertNotIn(config.obs_key, wrapper.observation_space.spaces)

        obs, reward, _, info = wrapper.step(np.zeros(1))
        self.assertNotIn(config.obs_key, obs)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info['stage_reward'], 0.0)
        np.testing.assert_array_equal(info['completed_stage_mask'], [1.0])

    def test_multistep_aggregates_task_reward_from_info(self):
        env = FakeSquareEnv([
            ((0.0, 0.35, 0.0, 0.0), 0.0),
            ((0.0, 0.0, 0.0, 0.0), 1.0),
            ((0.0, 0.0, 0.0, 0.0), 0.0),
        ])
        wrapper = SquareSubtaskWrapper(env, make_config(['grasp'], [2.0]))
        env = MultiStepWrapper(
            wrapper, n_obs_steps=1, n_action_steps=3, reward_agg_method='max')
        env.reset()

        _, reward, _, info = env.step(np.zeros((3, 1)))
        self.assertEqual(reward, 2.0)
        self.assertEqual(info['raw_reward'], 3.0)
        self.assertEqual(info['task_reward'], 1.0)

    def test_multistep_requires_task_reward_info(self):
        env = FakeSquareEnv([((0.0, 0.0, 0.0, 0.0), 1.0)])
        env = MultiStepWrapper(
            env, n_obs_steps=1, n_action_steps=1, reward_agg_method='max')
        env.reset()

        with self.assertRaises(KeyError):
            env.step(np.zeros((1, 1)))

    def test_pbrs_progress_holding_and_reset(self):
        env = FakeSquareEnv([
            ((0.0, 0.35, 0.0, 0.0), 0.0),
            ((0.0, 0.0, 0.0, 0.0), 0.0),
            ((0.0, 0.0, 0.0, 0.7), 0.0),
            ((0.0, 0.0, 0.0, 0.0), 0.0),
        ])
        config = make_config(['grasp', 'hover'], [0.5, 0.5], reward_mode='pbrs')
        config.reward_scale = 2.0
        wrapper = SquareSubtaskWrapper(env, config, gamma=0.9)
        wrapper.reset()

        for expected in (0.9, -0.1, 0.8, -0.2):
            obs, reward, _, info = wrapper.step(np.zeros(1))
            self.assertAlmostEqual(reward, expected)
            self.assertAlmostEqual(info['stage_reward'], expected)
            self.assertEqual(info['task_reward'], 0.0)
        np.testing.assert_array_equal(obs['completed_stage_mask'], [1.0, 1.0])
        np.testing.assert_array_equal(
            wrapper.reset()['completed_stage_mask'], [0.0, 0.0])

    def test_pbrs_terminal_return_matches_task_return(self):
        for task_reward in (0.0, 1.0):
            for complete_on_terminal in (False, True):
                with self.subTest(task_reward=task_reward,
                                  complete_on_terminal=complete_on_terminal):
                    grasp = (0.0, 0.35, 0.0, 0.0)
                    empty = (0.0, 0.0, 0.0, 0.0)
                    env = FakeSquareEnv([
                        (empty if complete_on_terminal else grasp, 0.0),
                        (empty, 0.0),
                        (grasp, task_reward, True, {}),
                    ])
                    wrapper = SquareSubtaskWrapper(
                        env, make_config(['grasp'], [0.5], reward_mode='pbrs'),
                        gamma=0.9)
                    wrapper.reset()

                    total = 0.0
                    for t in range(3):
                        obs, reward, done, info = wrapper.step(np.zeros(1))
                        total += 0.9 ** t * reward
                    self.assertTrue(done)
                    self.assertAlmostEqual(total, 0.9 ** 2 * task_reward)
                    self.assertEqual(info['task_reward'], task_reward)
                    np.testing.assert_array_equal(obs['completed_stage_mask'], [1.0])

    def test_pbrs_multistep_discount_matches_primitive_return(self):
        env = FakeSquareEnv([
            ((0.0, 0.35, 0.0, 0.0), 0.0),
            ((0.0, 0.35, 0.0, 0.7), 0.0),
            ((0.0, 0.0, 0.0, 0.0), 0.0),
            ((0.0, 0.0, 0.0, 0.0), 1.0, True, {}),
        ])
        wrapper = SquareSubtaskWrapper(
            env, make_config(['grasp', 'hover'], [0.5, 0.5], reward_mode='pbrs'),
            gamma=0.8)
        env = MultiStepWrapper(
            wrapper, n_obs_steps=1, n_action_steps=2,
            reward_agg_method='discounted_sum', gamma=0.8)
        env.reset()

        _, first, done, _ = env.step(np.zeros((2, 1)))
        self.assertFalse(done)
        self.assertAlmostEqual(first, 0.8 ** 2)
        _, second, done, info = env.step(np.zeros((2, 1)))
        self.assertTrue(done)
        self.assertAlmostEqual(first + 0.8 ** 2 * second, 0.8 ** 3)
        self.assertEqual(info['task_reward'], 1.0)

    def test_pbrs_timeout_keeps_potential_for_bootstrap(self):
        for inner_timeout in (False, True):
            with self.subTest(inner_timeout=inner_timeout):
                env = FakeSquareEnv([
                    ((0.0, 0.35, 0.0, 0.0), 0.0),
                    ((0.0, 0.0, 0.0, 0.0), 0.0, inner_timeout,
                     {'TimeLimit.truncated': True} if inner_timeout else {}),
                ])
                wrapper = SquareSubtaskWrapper(
                    env, make_config(['grasp'], [1.0], reward_mode='pbrs'),
                    gamma=0.8)
                env = MultiStepWrapper(
                    wrapper, n_obs_steps=1, n_action_steps=2,
                    max_episode_steps=None if inner_timeout else 2,
                    reward_agg_method='discounted_sum', gamma=0.8)
                env.reset()

                _, reward, done, info = env.step(np.zeros((2, 1)))
                self.assertTrue(done)
                self.assertTrue(info['TimeLimit.truncated'])
                self.assertAlmostEqual(reward, 0.8 ** 2)
                np.testing.assert_array_equal(
                    info['terminal_observation']['completed_stage_mask'], [[1.0]])
                # V_pbrs = V_task - Phi; retained potential cancels in the target.
                self.assertAlmostEqual(reward + 0.8 ** 2 * (2.0 - 1.0), 0.8 ** 2 * 2.0)

    def test_pbrs_requires_explicit_gamma(self):
        env = FakeSquareEnv([])
        with self.assertRaises(AssertionError):
            SquareSubtaskWrapper(
                env, make_config(['grasp'], [1.0], reward_mode='pbrs'))

    def test_timeout_preserves_terminal_checklist(self):
        env = FakeSquareEnv([((0.0, 0.35, 0.0, 0.0), 0.0)])
        env = MultiStepWrapper(
            SquareSubtaskWrapper(env, make_config(['grasp'], [1.0])),
            n_obs_steps=1,
            n_action_steps=1,
            max_episode_steps=1,
        )
        env.reset()

        _, _, done, info = env.step(np.zeros((1, 1)))

        self.assertEqual(done, 1.0)
        self.assertTrue(info['TimeLimit.truncated'])
        np.testing.assert_array_equal(
            info['terminal_observation']['completed_stage_mask'], [[1.0]])
        reset_obs = env.reset()
        np.testing.assert_array_equal(
            reset_obs['completed_stage_mask'], [[0.0]])


if __name__ == '__main__':
    unittest.main()
