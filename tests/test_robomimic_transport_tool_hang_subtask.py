import unittest
from types import SimpleNamespace

import gym
from gym import spaces
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path

from zprl.env.robomimic.robomimic_subtask_wrapper import (
    get_subtask_dim, make_subtask_wrapper,
)


class FakeTask:
    def __init__(self):
        self.transport = SimpleNamespace(
            trash_in_trash_bin=False, payload_in_target_bin=False)
        self.frame_assembled = False

    def _check_success(self):
        return self.transport.trash_in_trash_bin and self.transport.payload_in_target_bin

    def _check_frame_assembled(self):
        return self.frame_assembled


class FakeEnv(gym.Env):
    def __init__(self):
        self.env = FakeTask()
        self.observation_space = spaces.Dict({
            'state': spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)})
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.reward = 0.0
        self.done = False
        self.info = {}

    def reset(self):
        return {'state': np.zeros(1, dtype=np.float32)}

    def step(self, action):
        return self.reset(), self.reward, self.done, self.info.copy()


def make_wrapper(task='transport', mode='semi_sparse', scale=2.0, enabled=True):
    env = FakeEnv()
    stages = ['trash_in_trash_bin', 'payload_in_target_bin']
    name = 'TwoArmTransport'
    if task == 'tool_hang':
        stages = ['frame_assembled']
        name = 'ToolHang'
    config = SimpleNamespace(
        enabled=enabled, stages=stages, reward_mode=mode,
        reward_scale=scale, stage_weights=[0.5] * len(stages),
        obs_key='completed_stage_mask')
    wrapper = make_subtask_wrapper(env, config, name, gamma=0.9)
    wrapper.reset()
    return env, wrapper


class TransportToolHangSubtaskTest(unittest.TestCase):
    def test_transport_both_orders_and_success(self):
        for first, second in ((0, 1), (1, 0)):
            with self.subTest(first=first):
                env, wrapper = make_wrapper()
                stages = wrapper.stages
                setattr(env.env.transport, stages[first], True)
                obs, reward, _, info = wrapper.step(None)
                self.assertEqual(reward, 1.0)
                self.assertEqual(info['stage_reward'], 0.5)
                self.assertEqual(obs['completed_stage_mask'][first], 1.0)
                self.assertEqual(wrapper.step(None)[1], 0.0)
                setattr(env.env.transport, stages[second], True)
                env.reward, env.done = 1.0, True
                obs, reward, done, info = wrapper.step(None)
                self.assertTrue(done)
                self.assertEqual(reward, 1.0)
                self.assertEqual(info['stage_reward'], 0.0)
                self.assertEqual(info['stage_completion_delta'][second], 1.0)
                np.testing.assert_array_equal(obs['completed_stage_mask'], [1, 1])

    def test_transport_simultaneous_success_has_no_bonus(self):
        env, wrapper = make_wrapper()
        env.env.transport.trash_in_trash_bin = True
        env.env.transport.payload_in_target_bin = True
        env.reward = 1.0
        # Success need not terminate evaluation environments.
        _, reward, _, info = wrapper.step(None)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info['stage_reward'], 0.0)
        np.testing.assert_array_equal(info['stage_completion_delta'], [1, 1])

    def test_transport_history_is_not_current_success(self):
        env, wrapper = make_wrapper()
        env.env.transport.trash_in_trash_bin = True
        wrapper.step(None)
        env.env.transport.trash_in_trash_bin = False
        env.env.transport.payload_in_target_bin = True
        _, reward, _, info = wrapper.step(None)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info['task_reward'], 0.0)
        np.testing.assert_array_equal(info['completed_stage_mask'], [1, 1])
        np.testing.assert_array_equal(wrapper.reset()['completed_stage_mask'], [0, 0])

    def test_tool_hang_official_predicate_and_one_time_bonus(self):
        env, wrapper = make_wrapper('tool_hang')
        self.assertEqual(wrapper.step(None)[1], 0.0)
        env.env.frame_assembled = True
        _, reward, _, info = wrapper.step(None)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info['stage_reward'], 0.5)
        env.env.frame_assembled = False
        self.assertEqual(wrapper.step(None)[1], 0.0)
        env.env.frame_assembled = True
        env.reward, env.done = 1.0, True
        self.assertEqual(wrapper.step(None)[1], 1.0)

    def test_sparse_disabled_and_zero_scale(self):
        for task in ('transport', 'tool_hang'):
            for mode, scale, enabled in (
                    ('sparse', 2.0, True), ('semi_sparse', 0.0, True),
                    ('semi_sparse', 2.0, False)):
                with self.subTest(task=task, mode=mode, scale=scale, enabled=enabled):
                    env, wrapper = make_wrapper(task, mode, scale, enabled)
                    env.env.transport.trash_in_trash_bin = True
                    env.env.frame_assembled = True
                    obs, reward, _, info = wrapper.step(None)
                    self.assertEqual(reward, 0.0)
                    self.assertEqual('completed_stage_mask' in obs, enabled)
                    self.assertEqual(info['completed_stage_mask'][0], 1.0)
                    self.assertEqual(info['stage_reward'], 0.5 if scale == 0 else 0.0)

    def test_pbrs_terminal_correction_and_timeout(self):
        for task in ('transport', 'tool_hang'):
            for timeout in (False, True):
                with self.subTest(task=task, timeout=timeout):
                    env, wrapper = make_wrapper(task, 'pbrs')
                    env.env.transport.trash_in_trash_bin = True
                    env.env.frame_assembled = True
                    self.assertAlmostEqual(wrapper.step(None)[1], 0.9)
                    env.done = True
                    env.info = {'TimeLimit.truncated': timeout}
                    if not timeout:
                        env.env.transport.payload_in_target_bin = True
                        env.reward = 1.0
                    _, reward, _, info = wrapper.step(None)
                    self.assertAlmostEqual(info['stage_reward'], -0.05 if timeout else -0.5)
                    self.assertAlmostEqual(reward, -0.1 if timeout else 0.0)

    def test_no_subtask_preserves_observation_and_task_reward(self):
        env = FakeEnv()
        wrapper = make_subtask_wrapper(env, None, 'Can')
        env.reward = 1.0
        obs, reward, _, info = wrapper.step(None)
        self.assertEqual(get_subtask_dim(None), 0)
        self.assertNotIn('completed_stage_mask', obs)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info['task_reward'], 1.0)

    def test_task_configs_propagate_to_runner(self):
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        config_dir = str(Path(__file__).parents[1] / 'zprl' / 'config')
        for task, dim in (('transport', 2), ('tool_hang', 1)):
            with self.subTest(task=task), initialize_config_dir(
                    version_base=None, config_dir=config_dir):
                cfg = compose('train_online_robomimic_workspace', overrides=[
                    'online_task=' + task + '_image_abs', 'single_gamma=0.91'])
                self.assertEqual(get_subtask_dim(cfg.online_task.subtask), dim)
                self.assertEqual(cfg.online_task.env_runner.subtask, cfg.online_task.subtask)
                self.assertEqual(cfg.online_task.env_runner.gamma, 0.91)


if __name__ == '__main__':
    unittest.main()
