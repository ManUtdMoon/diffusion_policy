import gym
import numpy as np
from gym import spaces

from zprl.env.adroit.rrl_local.rrl_multicam import _close_render_context


class MetaWorldEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, device_id=0, rgb_size=84):
        super(MetaWorldEnv, self).__init__()
        import metaworld

        if '-v2' not in task_name:
            task_name = task_name + '-v2-goal-observable'

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        self.env._freeze_rand_vec = False

        self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5

        device_id_map = {
            0: 1,
            1: 0,
            2: 2,
            3: 3,
        }
        self.device_id = device_id_map[device_id]
        self.image_size = rgb_size

        self.episode_length = self._max_episode_steps = 200
        self.action_space = self.env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=1, shape=(3, self.image_size, self.image_size),
                dtype=np.float32),
            'agent_pos': spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_sensor_dim,),
                dtype=np.float32),
        })
        self.render_cache = None
        self._closed = False
        self._seed_pending = False

    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos('rightEndEffector'),
            self.env._get_site_pos('leftEndEffector')
        )
        return np.concatenate([eef_pos, finger_right, finger_left]).astype(
            np.float32)

    def get_rgb(self):
        return self.env.sim.render(
            width=self.image_size, height=self.image_size,
            camera_name="corner2", device_id=self.device_id)

    def render_high_res(self, resolution=512):
        return self.env.sim.render(
            width=resolution, height=resolution, camera_name="corner2",
            device_id=self.device_id)

    def _get_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)
        obs_dict = {
            'image': obs_pixels.astype(np.float32) / 255.0,
            'agent_pos': robot_state,
        }
        self.render_cache = np.moveaxis(obs_pixels, 0, -1)
        return obs_dict

    def step(self, action: np.array):
        _, _, _, env_info = self.env.step(action)
        self.cur_step += 1
        obs_dict = self._get_obs()

        done = False
        if self.cur_step >= self.episode_length:
            done = True
            env_info['TimeLimit.truncated'] = True

        reward = 1.0 if env_info['success'] else 0.0
        return obs_dict, reward, done, env_info

    def reset(self):
        if self._seed_pending:
            # First apply the seed to any model fields left by construction or
            # the preceding episode, then replay the same seeded reset.
            self.env.reset()
            self.env.seed(self._seed)
            self._seed_pending = False
        self.env.reset()
        self.cur_step = 0
        return self._get_obs()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        self.env.seeded_rand_vec = True
        self._seed_pending = True
        return self.env.seed(seed)

    def set_seed(self, seed=None):
        return self.seed(seed)

    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array'
        return self.render_cache.copy()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.env.close()
        finally:
            _close_render_context(self.env.sim)


class MetaworldEarlyStopWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if info.get('success', False):
            done = True
            info = info.copy()
            info.pop('TimeLimit.truncated', None)

        return obs, reward, done, info
