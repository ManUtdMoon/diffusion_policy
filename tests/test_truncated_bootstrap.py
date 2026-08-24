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
            self.assertIn(
                'bootstrap_at_done: truncated  # never, truncated', source, name)
            self.assertNotIn('always', source, name)


if __name__ == '__main__':
    unittest.main()
