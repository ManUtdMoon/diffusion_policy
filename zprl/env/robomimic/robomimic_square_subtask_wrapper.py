import gym
from gym import spaces
import numpy as np


def get_subtask_dim(subtask_config):
    if not subtask_config.enabled:
        return 0
    return len(subtask_config.stages)


class SquareSubtaskWrapper(gym.Wrapper):
    def __init__(self, env, subtask_config):
        super().__init__(env)

        assert isinstance(env.observation_space, spaces.Dict)

        self.enabled = subtask_config.enabled
        self.stages = tuple(subtask_config.stages)
        self.reward_mode = subtask_config.reward_mode
        self.reward_scale = float(subtask_config.reward_scale)
        self.stage_weights = np.asarray(
            subtask_config.stage_weights, dtype=np.float32)
        self.hover_threshold = float(subtask_config.hover_threshold)
        self.obs_key = subtask_config.obs_key

        assert self.stages in (('grasp',), ('grasp', 'hover'))
        assert self.reward_mode in ('sparse', 'semi_sparse')
        assert self.stage_weights.shape == (len(self.stages),)

        self.task = env
        while not hasattr(self.task, 'staged_rewards'):
            self.task = self.task.env
        assert self.task.single_object_mode == 2
        assert self.task.nut_id == self.task.nut_to_id['square']

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
        obs, task_reward, done, info = self.env.step(action)
        predicates = self._get_stage_predicates()
        completion_delta = self._commit_stages(predicates)

        stage_reward = 0.0
        if self.enabled and self.reward_mode == 'semi_sparse':
            stage_reward = self.reward_scale * float(
                np.dot(self.stage_weights, completion_delta))
        reward = task_reward + stage_reward

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
        _, r_grasp, _, r_hover = self.task.staged_rewards()
        predicate_map = {
            'grasp': r_grasp > 0.0,
            'hover': r_hover >= self.hover_threshold,
        }
        return np.asarray([predicate_map[stage] for stage in self.stages], dtype=np.bool_)

    def _commit_stages(self, predicates):
        completion_delta = np.logical_and(
            predicates, self.completed_stage_mask == 0.0)
        self.completed_stage_mask[completion_delta] = 1.0
        return completion_delta.astype(np.float32)
