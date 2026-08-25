# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

import gym
import numpy as np
from collections import deque


_mj_envs = {'pen-v0', 'hammer-v0', 'door-v0'}


def _close_render_context(sim):
    if sim is None:
        return
    for name in ('_render_context_offscreen', '_render_context_window'):
        render_context = getattr(sim, name, None)
        opengl_context = getattr(render_context, 'opengl_context', None)
        close = getattr(opengl_context, 'close', None)
        if close is not None:
            close()


class BasicAdroitEnv(gym.Env):
    def __init__(self, env, cameras, latent_dim=512, hybrid_state=True,
            channels_first=False, height=84, width=84, test_image=False,
            num_repeats=1, num_frames=1, device=None,
            render_device_id=0):
        self._env = env
        self.env_id = env.env.unwrapped.spec.id
        self.device = device
        render_id_map = {
            0: 1,
            1: 0,
            2: 2,
            3: 3,
        }
        self.render_device_id = render_id_map.get(render_device_id, 0)

        self._num_repeats = num_repeats
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)

        self.test_image = test_image
        self.cameras = cameras
        self.latent_dim = latent_dim
        self.hybrid_state = hybrid_state
        self.channels_first = channels_first
        self.height = height
        self.width = width
        self.action_space = self._env.action_space
        self.env_kwargs = {'cameras' : cameras, 'latent_dim' : latent_dim,
                           'hybrid_state': hybrid_state,
                           'channels_first' : channels_first, 'height' : height,
                           'width' : width}

        shape = [3, self.width, self.height]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        self.sim = env.env.sim
        self._env.spec.observation_dim = latent_dim

        if hybrid_state and self.env_id in _mj_envs:
            self._env.spec.observation_dim += 24

        self.spec = self._env.spec
        self.observation_dim = self.spec.observation_dim
        self.horizon = self._env.env.spec.max_episode_steps
        self._closed = False

    def get_obs(self):
        env_state = self._env.env.get_env_state()
        qp = env_state['qpos']

        if self.env_id == 'pen-v0':
            qp = qp[:-6]
        elif self.env_id == 'door-v0':
            qp = qp[4:-2]
        elif self.env_id == 'hammer-v0':
            qp = qp[2:-7]

        imgs = []
        if not self.test_image:
            for cam in self.cameras:
                img = self._env.env.sim.render(
                    width=self.width, height=self.height, mode='offscreen',
                    camera_name=cam, device_id=self.render_device_id)
                if self.channels_first:
                    img = img.transpose((2, 0, 1))
                imgs.append(img)
        else:
            img = (np.random.rand(1, self.height, self.width) * 255).astype(np.uint8)
            imgs.append(img)
        pixels = np.concatenate(imgs, axis=0)

        if not self.hybrid_state:
            qp = None
        return pixels, qp

    def set_seed(self, seed):
        return self._env.set_seed(seed)

    def get_stacked_pixels(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=0)

    def reset(self):
        self._env.reset()
        pixels, sensor_info = self.get_obs()
        self._frames.clear()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self.get_stacked_pixels(), sensor_info

    def step(self, action):
        reward_sum = 0.0
        n_goal_achieved = 0
        for _ in range(self._num_repeats):
            _, reward, done, env_info = self._env.step(action)
            reward_sum += reward
            if env_info['goal_achieved']:
                n_goal_achieved += 1
            if done:
                break
        env_info['n_goal_achieved'] = n_goal_achieved
        pixels, sensor_info = self.get_obs()
        self._frames.append(pixels)
        return [self.get_stacked_pixels(), sensor_info], reward_sum, done, env_info

    def set_env_state(self, state):
        return self._env.set_env_state(state)

    def get_env_state(self):
        return self._env.get_env_state()

    def get_pixels_with_width_height(self, w, h):
        imgs = []
        for cam in self.cameras:
            img = self._env.env.sim.render(
                width=w, height=h, mode='offscreen', camera_name=cam,
                device_id=self.render_device_id)
            if self.channels_first:
                img = img.transpose((2, 0, 1))
            imgs.append(img)
        return np.concatenate(imgs, axis=0)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._env.env, 'close', None)
            if close is not None:
                close()
        finally:
            _close_render_context(self.sim)
