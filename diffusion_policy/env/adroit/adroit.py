# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


from collections import deque
import warnings
import numpy as np
from dm_env import StepType, specs
from gym import spaces
import mj_envs  # ensure Adroit tasks (e.g., door-v0) register with Gym
from mjrl.utils.gym_env import GymEnv

from diffusion_policy.env.adroit.rrl_local.rrl_multicam import BasicAdroitEnv


class AdroitEnv:
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    # a wrapper class that will make Adroit env looks like a dmc env
    def __init__(self, env_name, test_image=False, cam_list=None,
                 num_repeats=2, num_frames=1, env_feature_type='pixels', device='cuda', render_device_id=0, reward_rescale=True):
        if '-v0' not in env_name:  # compatibility with gym env name
            env_name += '-v0'
        default_env_to_cam_list = {
            'hammer-v0': ['top'],
            'door-v0': ['top'],
            'pen-v0': ['vil_camera'],
            # 'relocate-v0': ['cam1', 'cam2', 'cam3',],
            'relocate-v0': ['cam1',],
        }
        if cam_list is None:
            cam_list = default_env_to_cam_list[env_name]
        self.env_name = env_name
        reward_rescale_dict = {
            'hammer-v0': 1/100,
            'door-v0': 1/20,
            'pen-v0': 1/50,
            'relocate-v0': 1/30,
        }
        if reward_rescale:
            self.reward_rescale_factor = reward_rescale_dict[env_name]
        else:
            self.reward_rescale_factor = 1

        env = GymEnv(env_name)

        assert env_feature_type == 'pixels'
        height = 84
        width = 84
        latent_dim = height * width * len(cam_list) * num_frames
        # RRL class instance is environment wrapper...
        env = BasicAdroitEnv(env, cameras=cam_list,
                                height=height, width=width, latent_dim=latent_dim, hybrid_state=True,
                                test_image=test_image, channels_first=True, num_repeats=num_repeats, num_frames=num_frames, device=device, render_device_id=render_device_id)

        self._env = env
        self.obs_dim = env.spec.observation_dim
        self.obs_sensor_dim = 24
        self.act_dim = env.spec.action_dim
        self.horizon = env.spec.horizon
        number_channel = len(cam_list) * 3 * num_frames

        if env_feature_type == 'pixels':
            self._obs_spec = specs.BoundedArray(shape=(
                number_channel, 84, 84), dtype='uint8', name='observation', minimum=0, maximum=255)
            self._obs_sensor_spec = specs.Array(
                shape=(self.obs_sensor_dim,), dtype='float32', name='observation_sensor')

        self._action_spec = specs.BoundedArray(shape=(
            self.act_dim,), dtype='float32', name='action', minimum=-1.0, maximum=1.0)

        # for diffusion policy codebase
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_dim,),
            dtype=np.float64
        )
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(number_channel, 84, 84),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
        })

    def reset(self):
        # pixels and sensor values
        obs_pixels, obs_sensor = self._env.reset()
        obs_sensor = obs_sensor.astype(np.float32)
        action_spec = self.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        obs_dict = {
            'image': obs_pixels,
            'agent_pos': obs_sensor
        }
        return obs_dict

    def get_pixels_with_width_height(self, w, h):
        return self._env.get_pixels_with_width_height(w, h)

    def step(self, action, force_step_type=None, debug=False):

        obs_all, reward, done, env_info = self._env.step(action)

        obs_pixels, obs_sensor = obs_all
        obs_sensor = obs_sensor.astype(np.float32)

        discount = 1.0
        n_goal_achieved = env_info['n_goal_achieved']
        time_limit_reached = env_info['TimeLimit.truncated'] if 'TimeLimit.truncated' in env_info else False
        if done:
            steptype = StepType.LAST
        else:
            steptype = StepType.MID

        if done and not time_limit_reached:
            discount = 0.0

        reward = reward * self.reward_rescale_factor

        obs_dict = {
            'image': obs_pixels,  # (3, 84, 84), [0,255], uint8
            'agent_pos': obs_sensor  # (24,)
        }

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
        pass

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def render(self, mode):
        assert mode == 'rgb_array'
        img = self.get_pixels_with_width_height(84, 84)
        # make it channel last
        img = np.transpose(img, (1, 2, 0))  # it has been 0-255
        # (84, 84, 3), uint8, 0-255
        return img

    def get_mujoco_sim(self):
        """
        return the underlying mujoco sim
        """
        return self._env.sim



if __name__ == "__main__":
    env = AdroitEnv('door')
    obs = env.reset()
    for _ in range(300):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        img = env.render(mode='rgb_array')
        if done:
            obs = env.reset()
    
    env = AdroitEnv('pen')
    obs = env.reset()
    for _ in range(300):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        img = env.render(mode='rgb_array')
        if done:
            obs = env.reset()
    input("Press Enter to continue...")