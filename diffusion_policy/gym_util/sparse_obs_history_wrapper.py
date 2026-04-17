import gym
from gym import spaces
import numpy as np


def _select_obs_steps(obs, indices):
    if isinstance(obs, np.ndarray):
        return obs[indices]
    if isinstance(obs, dict):
        return {key: _select_obs_steps(value, indices) for key, value in obs.items()}
    raise RuntimeError(f'Unsupported observation type {type(obs)}')


def _select_space_steps(space, indices):
    if isinstance(space, spaces.Box):
        return spaces.Box(
            low=space.low[indices],
            high=space.high[indices],
            shape=(len(indices),) + space.shape[1:],
            dtype=space.dtype,
        )
    if isinstance(space, spaces.Dict):
        result = spaces.Dict()
        for key, value in space.spaces.items():
            result[key] = _select_space_steps(value, indices)
        return result
    raise RuntimeError(f'Unsupported observation space type {type(space)}')


class SparseObsHistoryWrapper(gym.Wrapper):
    def __init__(self, env, obs_step_indices):
        super().__init__(env)
        self.obs_step_indices = np.array(obs_step_indices, dtype=np.int64)
        if self.obs_step_indices.ndim != 1 or len(self.obs_step_indices) == 0:
            raise ValueError('obs_step_indices must be a non-empty 1D list.')
        if np.any(self.obs_step_indices < 0):
            raise ValueError('obs_step_indices must be non-negative.')
        if np.any(np.diff(self.obs_step_indices) < 0):
            raise ValueError('obs_step_indices must be sorted in ascending order.')

        base_obs_space = env.observation_space
        if not isinstance(base_obs_space, (spaces.Box, spaces.Dict)):
            raise RuntimeError(f'Unsupported observation space type {type(base_obs_space)}')
        if self.obs_step_indices[-1] >= base_obs_space.shape[0] if isinstance(base_obs_space, spaces.Box) else False:
            raise ValueError('obs_step_indices exceeds Box observation history length.')
        if isinstance(base_obs_space, spaces.Dict):
            sample_key = next(iter(base_obs_space.spaces.keys()))
            sample_space = base_obs_space.spaces[sample_key]
            if self.obs_step_indices[-1] >= sample_space.shape[0]:
                raise ValueError('obs_step_indices exceeds Dict observation history length.')

        self._observation_space = _select_space_steps(base_obs_space, self.obs_step_indices)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return _select_obs_steps(obs, self.obs_step_indices)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _select_obs_steps(obs, self.obs_step_indices), reward, done, info
