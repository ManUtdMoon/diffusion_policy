from types import SimpleNamespace

import gym
from gym import spaces
import numpy as np


def get_subtask_dim(subtask_config):
    if subtask_config is None or not subtask_config.enabled:
        return 0
    return len(subtask_config.stages)


class SubtaskWrapper(gym.Wrapper):
    def __init__(self, env, subtask_config, gamma=None):
        super().__init__(env)

        assert isinstance(env.observation_space, spaces.Dict)

        if subtask_config is None:
            subtask_config = SimpleNamespace(
                enabled=False, stages=(), reward_mode='sparse',
                reward_scale=0.0, stage_weights=(), obs_key='completed_stage_mask')

        self.enabled = subtask_config.enabled
        self.stages = tuple(subtask_config.stages)
        self.reward_mode = subtask_config.reward_mode
        self.gamma = gamma
        self.reward_scale = float(subtask_config.reward_scale)
        self.stage_weights = np.asarray(
            subtask_config.stage_weights, dtype=np.float32)
        self.obs_key = subtask_config.obs_key

        assert self.reward_mode in ('sparse', 'semi_sparse', 'pbrs')
        if self.enabled and self.reward_mode == 'pbrs':
            assert gamma is not None and 0.0 <= gamma <= 1.0
        assert self.stage_weights.shape == (len(self.stages),)

        if self.enabled:
            assert self.obs_key not in env.observation_space.spaces
            obs_spaces = dict(env.observation_space.spaces)
            obs_spaces[self.obs_key] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self.stages),),
                dtype=np.float32
            )
            self.observation_space = spaces.Dict(obs_spaces)
        self.completed_stage_mask = np.zeros(len(self.stages), dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.completed_stage_mask.fill(0.0)
        return self._augment_observation(obs)

    def step(self, action):
        phi = float(np.dot(self.stage_weights, self.completed_stage_mask))
        obs, task_reward, done, info = self.env.step(action)
        predicates = self._get_stage_predicates()
        completion_delta = self._commit_stages(predicates)

        stage_reward = 0.0
        if self.enabled and self.reward_mode == 'semi_sparse':
            stage_reward = self._stage_bonus(completion_delta)
        elif self.enabled and self.reward_mode == 'pbrs':
            next_phi = float(np.dot(self.stage_weights, self.completed_stage_mask))
            if done and not info.get('TimeLimit.truncated', False):
                next_phi = 0.0
            stage_reward = self.gamma * next_phi - phi
        reward = task_reward + self.reward_scale * stage_reward

        info = info.copy()
        info['completed_stage_mask'] = self.completed_stage_mask.copy()
        info['stage_completion_delta'] = completion_delta
        info['task_reward'] = float(task_reward)
        info['stage_reward'] = stage_reward
        return self._augment_observation(obs), reward, done, info

    def _augment_observation(self, obs):
        if not self.enabled:
            return obs
        result = dict(obs)
        result[self.obs_key] = self.completed_stage_mask.copy()
        return result

    def _get_stage_predicates(self):
        return np.zeros(len(self.stages), dtype=np.bool_)

    def _stage_bonus(self, completion_delta):
        return float(np.dot(self.stage_weights, completion_delta))

    def _commit_stages(self, predicates):
        completion_delta = np.logical_and(
            predicates, self.completed_stage_mask == 0.0)
        self.completed_stage_mask[completion_delta] = 1.0
        return completion_delta.astype(np.float32)


def make_subtask_wrapper(env, subtask_config, env_name, gamma=None):
    if subtask_config is None:
        return SubtaskWrapper(env, None, gamma=gamma)

    from zprl.env.robomimic.robomimic_square_subtask_wrapper import SquareSubtaskWrapper
    from zprl.env.robomimic.robomimic_transport_subtask_wrapper import TransportSubtaskWrapper
    from zprl.env.robomimic.robomimic_tool_hang_subtask_wrapper import ToolHangSubtaskWrapper

    wrappers = {
        'NutAssemblySquare': SquareSubtaskWrapper,
        'TwoArmTransport': TransportSubtaskWrapper,
        'ToolHang': ToolHangSubtaskWrapper,
    }
    return wrappers[env_name](env, subtask_config, gamma=gamma)
