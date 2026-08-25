# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from dm_env import specs
import gym
from gym import spaces

from zprl.env.adroit.rrl_local.rrl_multicam import BasicAdroitEnv


class AdroitEnv:
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, env_name, test_image=False, cam_list=None,
                 num_repeats=2, num_frames=1, env_feature_type='pixels',
                 device='cuda', render_device_id=0, reward_rescale=True):
        import mj_envs  # noqa: F401; register Adroit tasks with Gym
        from mjrl.utils.gym_env import GymEnv

        if '-v0' not in env_name:
            env_name += '-v0'
        default_env_to_cam_list = {
            'hammer-v0': ['top'],
            'door-v0': ['top'],
            'pen-v0': ['vil_camera'],
        }
        if cam_list is None:
            cam_list = default_env_to_cam_list[env_name]
        self.env_name = env_name
        reward_rescale_dict = {
            'hammer-v0': 1/100,
            'door-v0': 1/20,
            'pen-v0': 1/50,
        }
        self.reward_rescale_factor = (
            reward_rescale_dict[env_name] if reward_rescale else 1)

        env = GymEnv(env_name)

        assert env_feature_type == 'pixels'
        height = 84
        width = 84
        latent_dim = height * width * len(cam_list) * num_frames
        env = BasicAdroitEnv(
            env, cameras=cam_list, height=height, width=width,
            latent_dim=latent_dim, hybrid_state=True, test_image=test_image,
            channels_first=True, num_repeats=num_repeats,
            num_frames=num_frames, device=device,
            render_device_id=render_device_id)

        self._env = env
        self.obs_dim = env.spec.observation_dim
        self.obs_sensor_dim = 24
        self.act_dim = env.spec.action_dim
        self.horizon = env.spec.horizon
        number_channel = len(cam_list) * 3 * num_frames

        self._obs_spec = specs.BoundedArray(
            shape=(number_channel, 84, 84), dtype='uint8', name='observation',
            minimum=0, maximum=255)
        self._obs_sensor_spec = specs.Array(
            shape=(self.obs_sensor_dim,), dtype='float32',
            name='observation_sensor')
        self._action_spec = specs.BoundedArray(
            shape=(self.act_dim,), dtype='float32', name='action',
            minimum=-1.0, maximum=1.0)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float64)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=1, shape=(number_channel, 84, 84),
                dtype=np.float32),
            'agent_pos': spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_sensor_dim,),
                dtype=np.float32),
        })
        self.render_cache = None
        self._closed = False

    def reset(self):
        obs_pixels, obs_sensor = self._env.reset()
        obs_dict = {
            'image': obs_pixels.astype(np.float32) / 255.0,
            'agent_pos': obs_sensor.astype(np.float32),
        }
        self.render_cache = np.moveaxis(obs_pixels, 0, -1)
        return obs_dict

    def get_pixels_with_width_height(self, w, h):
        return self._env.get_pixels_with_width_height(w, h)

    def step(self, action, force_step_type=None, debug=False):
        obs_all, reward, done, env_info = self._env.step(action)
        obs_pixels, obs_sensor = obs_all
        reward = reward * self.reward_rescale_factor
        obs_dict = {
            'image': obs_pixels.astype(np.float32) / 255.0,
            'agent_pos': obs_sensor.astype(np.float32),
        }
        self.render_cache = np.moveaxis(obs_pixels, 0, -1)
        return obs_dict, reward, done, env_info

    def observation_spec(self):
        return self._obs_spec

    def observation_sensor_spec(self):
        return self._obs_sensor_spec

    def action_spec(self):
        return self._action_spec

    def set_env_state(self, state):
        self._env.set_env_state(state)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._env.close()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        return self._env.set_seed(seed)

    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array'
        return self.render_cache.copy()

    def get_mujoco_sim(self):
        return self._env.sim


class AdroitEarlyStopWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if 'door_pos' in info:
            done = True if info['door_pos'] < -0.1 else done

        if 'drop' in info:
            done = True if info['drop'] else done

        return obs, reward, done, info
