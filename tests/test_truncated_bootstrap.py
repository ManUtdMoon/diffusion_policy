import pathlib
import unittest

import gym
import numpy as np

from zprl.gym_util.multistep_wrapper import MultiStepWrapper


class CountingEnv(gym.Env):
    metadata = {}

    def __init__(self, terminate_at=None, truncate_at=None):
        self.terminate_at = terminate_at
        self.truncate_at = truncate_at
        self.step_count = 0
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self):
        self.step_count = 0
        return {'state': np.array([0.0], dtype=np.float32)}

    def step(self, action):
        self.step_count += 1
        terminated = self.step_count == self.terminate_at
        truncated = self.step_count == self.truncate_at
        info = {'TimeLimit.truncated': True} if truncated else {}
        obs = {'state': np.array([self.step_count], dtype=np.float32)}
        return obs, 0.0, terminated or truncated, info


class RewardEnv(CountingEnv):
    def step(self, action):
        obs, _, done, info = super().step(action)
        return obs, 2.0, done, info


class SuccessAndTimeoutEnv(CountingEnv):
    def step(self, action):
        obs, _, _, _ = super().step(action)
        return obs, 1.0, True, {
            'success': True, 'TimeLimit.truncated': True}


class TruncatedBootstrapTest(unittest.TestCase):
    def test_wrapper_preserves_true_timeout_observation(self):
        env = MultiStepWrapper(
            CountingEnv(), n_obs_steps=2, n_action_steps=3,
            max_episode_steps=2)
        env.reset()

        obs, _, done, info = env.step(np.zeros((3, 1), dtype=np.float32))

        self.assertTrue(done)
        self.assertTrue(info['TimeLimit.truncated'])
        np.testing.assert_array_equal(obs['state'], [[1.0], [2.0]])
        np.testing.assert_array_equal(
            info['terminal_observation']['state'], [[1.0], [2.0]])
        self.assertEqual(env.env.step_count, 2)

    def test_true_termination_is_not_labeled_timeout(self):
        env = MultiStepWrapper(
            CountingEnv(terminate_at=2), n_obs_steps=2, n_action_steps=3,
            max_episode_steps=2)
        env.reset()

        _, _, done, info = env.step(np.zeros((3, 1), dtype=np.float32))

        self.assertTrue(done)
        self.assertNotIn('TimeLimit.truncated', info)
        self.assertNotIn('terminal_observation', info)
        self.assertEqual(env.env.step_count, 2)

    def test_target_workspaces_bootstrap_only_timeouts(self):
        root = pathlib.Path(__file__).parents[1]
        workspace_names = [
            'train_online_vib_robomimic_workspace.py',
            'train_online_noise_robomimic_workspace.py',
            'train_online_robomimic_workspace.py',
        ]
        for name in workspace_names:
            source = (root / 'zprl' / 'workspace' / name).read_text()
            self.assertIn(
                'if bootstrap_at_done == \'truncated\':', source, name)
            self.assertIn(
                "if info.get('TimeLimit.truncated', False)", source, name)
            self.assertIn(
                "terminal_observation = infos[i]['terminal_observation']",
                source, name)
            self.assertIn(
                'rb_dones[timeout_indices] = False', source, name)
            self.assertIn('done=rb_dones', source, name)

        config_names = [name.replace('.py', '.yaml') for name in workspace_names]
        for name in config_names:
            source = (root / 'zprl' / 'config' / name).read_text()
            self.assertRegex(
                source, r"bootstrap_at_done: (never|truncated)", name)
            self.assertNotIn("always", source, name)


    def test_success_is_not_mislabeled_as_timeout(self):
        from zprl.env.metaworld.metaworld_image_wrapper import (
            MetaworldEarlyStopWrapper)

        env = MultiStepWrapper(
            MetaworldEarlyStopWrapper(SuccessAndTimeoutEnv()),
            n_obs_steps=1, n_action_steps=3, max_episode_steps=1)
        env.reset()

        _, _, done, info = env.step(np.zeros((3, 1), dtype=np.float32))

        self.assertTrue(done)
        self.assertNotIn('TimeLimit.truncated', info)
        self.assertNotIn('terminal_observation', info)
        self.assertEqual(env.env.env.step_count, 1)

    def test_reward_offset_preserves_raw_reward(self):
        env = MultiStepWrapper(
            RewardEnv(), n_obs_steps=1, n_action_steps=2,
            max_episode_steps=3, reward_agg_method='discounted_sum',
            gamma=0.5, reward_offset=-1.0)
        env.reset()

        _, reward, done, info = env.step(
            np.zeros((2, 1), dtype=np.float32))

        self.assertFalse(done)
        self.assertAlmostEqual(reward, 1.5)
        self.assertAlmostEqual(info['raw_reward'], 2.0)

    def test_chunk_stops_on_last_low_level_termination(self):
        env = MultiStepWrapper(
            CountingEnv(terminate_at=3), n_obs_steps=2, n_action_steps=3,
            max_episode_steps=3)
        env.reset()

        obs, _, done, info = env.step(np.zeros((3, 1), dtype=np.float32))

        self.assertTrue(done)
        self.assertNotIn('TimeLimit.truncated', info)
        np.testing.assert_array_equal(obs['state'], [[2.0], [3.0]])
        self.assertEqual(env.env.step_count, 3)

    def test_chunk_stops_on_last_low_level_timeout(self):
        env = MultiStepWrapper(
            CountingEnv(), n_obs_steps=2, n_action_steps=3,
            max_episode_steps=3)
        env.reset()

        obs, _, done, info = env.step(np.zeros((3, 1), dtype=np.float32))

        self.assertTrue(done)
        self.assertTrue(info['TimeLimit.truncated'])
        np.testing.assert_array_equal(obs['state'], [[2.0], [3.0]])
        np.testing.assert_array_equal(
            info['terminal_observation']['state'], [[2.0], [3.0]])
        self.assertEqual(env.env.step_count, 3)

    def test_chunk_stops_on_first_low_level_termination(self):
        env = MultiStepWrapper(
            CountingEnv(terminate_at=1), n_obs_steps=2, n_action_steps=3,
            max_episode_steps=3)
        env.reset()

        obs, _, done, info = env.step(np.zeros((3, 1), dtype=np.float32))

        self.assertTrue(done)
        self.assertNotIn('TimeLimit.truncated', info)
        np.testing.assert_array_equal(obs['state'], [[0.0], [1.0]])
        self.assertEqual(env.env.step_count, 1)

    def test_chunk_stops_on_first_low_level_timeout(self):
        env = MultiStepWrapper(
            CountingEnv(), n_obs_steps=2, n_action_steps=3,
            max_episode_steps=1)
        env.reset()

        obs, _, done, info = env.step(np.zeros((3, 1), dtype=np.float32))

        self.assertTrue(done)
        self.assertTrue(info['TimeLimit.truncated'])
        np.testing.assert_array_equal(obs['state'], [[0.0], [1.0]])
        np.testing.assert_array_equal(
            info['terminal_observation']['state'], [[0.0], [1.0]])
        self.assertEqual(env.env.step_count, 1)

    def test_workspace_replay_uses_timeout_observation_only(self):
        from zprl.workspace import train_online_workspace
        from zprl.workspace import train_online_vib_workspace
        from zprl.workspace import train_online_noise_workspace

        helpers = [
            train_online_workspace._prepare_replay_next_obs,
            train_online_vib_workspace._prepare_replay_next_obs,
            train_online_noise_workspace._prepare_replay_next_obs,
        ]
        auto_reset_obs = {
            'state': np.array([[[-1.0], [-1.0]]], dtype=np.float32)
        }
        timeout_obs = {
            'state': np.array([[0.0], [1.0]], dtype=np.float32)
        }
        timeout_info = [{
            'TimeLimit.truncated': True,
            'terminal_observation': timeout_obs,
        }]
        termination_info = [{}]

        for helper in helpers:
            rb_obs, rb_done = helper(
                auto_reset_obs, np.array([True]), timeout_info, 'truncated')
            self.assertFalse(rb_done[0])
            np.testing.assert_array_equal(
                rb_obs['state'][0], timeout_obs['state'])

            rb_obs, rb_done = helper(
                auto_reset_obs, np.array([True]), termination_info, 'truncated')
            self.assertIs(rb_obs, auto_reset_obs)
            self.assertTrue(rb_done[0])

    def test_new_resrl_workspace_uses_full_Do_slice(self):
        import ast

        root = pathlib.Path(__file__).parents[1]
        path = root / 'zprl/workspace/train_online_workspace.py'
        tree = ast.parse(path.read_text())
        loaded_names = [
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        ]
        self.assertNotIn('obs_emb_dim', loaded_names)
        self.assertGreaterEqual(loaded_names.count('Do'), 4)


if __name__ == '__main__':
    unittest.main()
