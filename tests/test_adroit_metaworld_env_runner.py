from types import SimpleNamespace

import gym
import numpy as np

from zprl.env.adroit.adroit import AdroitEnv
from zprl.env.adroit.rrl_local.rrl_multicam import BasicAdroitEnv
from zprl.env.metaworld.metaworld_image_wrapper import MetaWorldEnv
from zprl.env_runner.adroit_runner import AdroitRunner
from zprl.env_runner.metaworld_runner import _make_init_fn
from zprl.gym_util.video_recording_wrapper import VideoRecordingWrapper


class FakeRecorder:
    def __init__(self):
        self.stop_count = 0

    def stop(self):
        self.stop_count += 1

    def is_ready(self):
        return False


class FakeGymEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32)
        self.close_count = 0

    def close(self):
        self.close_count += 1


class FakeContext:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class FakeSim:
    def __init__(self):
        self.device_ids = []
        self.context = FakeContext()
        self._render_context_offscreen = SimpleNamespace(
            opengl_context=self.context)
        self._render_context_window = None

    def render(self, width, height, **kwargs):
        self.device_ids.append(kwargs['device_id'])
        return np.zeros((height, width, 3), dtype=np.uint8)


class FakeAdroitGymEnv:
    def __init__(self):
        self.sim = FakeSim()
        self.spec = SimpleNamespace(id='door-v0', max_episode_steps=100)
        self.unwrapped = self
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32)
        self.close_count = 0

    def get_env_state(self):
        return {'qpos': np.arange(30)}

    def close(self):
        self.close_count += 1


class FakeGymEnvAdapter:
    def __init__(self):
        self.env = FakeAdroitGymEnv()
        self.action_space = self.env.action_space
        self.spec = SimpleNamespace(
            observation_dim=30, action_dim=2, horizon=100)
        self.seeds = []

    def set_seed(self, seed):
        self.seeds.append(seed)
        return [seed]


class FakeMetaWorldEnv:
    def __init__(self):
        self.seeded_rand_vec = False
        self.np_random = np.random.default_rng()
        self.state = None
        self.reset_count = 0

    def seed(self, seed):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def reset(self):
        self.reset_count += 1
        self.state = self.np_random.uniform(size=3)


def test_adroit_seed_forwarding_and_render_device_mapping():
    adapter = FakeGymEnvAdapter()
    env = BasicAdroitEnv(
        adapter, cameras=['top'], channels_first=True, render_device_id=0)
    env.get_obs()
    assert adapter.env.sim.device_ids == [1]

    wrapper = AdroitEnv.__new__(AdroitEnv)
    wrapper._env = adapter
    assert wrapper.seed(123) == [123]
    assert adapter.seeds == [123]


def test_metaworld_seed_controls_reset_and_render_device():
    raw_env = FakeMetaWorldEnv()
    env = MetaWorldEnv.__new__(MetaWorldEnv)
    env.env = raw_env
    env.cur_step = 0
    env.device_id = 2
    env.image_size = 84
    env._get_obs = lambda: {'state': raw_env.state.copy()}

    env.seed(7)
    first = env.reset()['state']
    next_unseeded = env.reset()['state']
    env.seed(7)
    second = env.reset()['state']
    env.seed(8)
    third = env.reset()['state']

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, next_unseeded)
    assert not np.array_equal(first, third)
    assert raw_env.seeded_rand_vec is True
    assert raw_env.reset_count == 7

    sim = FakeSim()
    env.env = SimpleNamespace(sim=sim)
    env.get_rgb()
    assert sim.device_ids == [2]


def test_metaworld_video_init_filenames_are_bound(tmp_path):
    recorder = FakeRecorder()
    wrapped = VideoRecordingWrapper(FakeGymEnv(), recorder)
    env = SimpleNamespace(env=wrapped, seeds=[])
    env.seed = env.seeds.append

    first = _make_init_fn(tmp_path, 0, True, 10000)
    second = _make_init_fn(tmp_path, 1, True, 10001)
    first(env)
    first_path = wrapped.file_path
    second(env)
    second_path = wrapped.file_path

    assert first_path.endswith('test_0.mp4')
    assert second_path.endswith('test_1.mp4')
    assert first_path != second_path
    assert env.seeds == [10000, 10001]


def test_close_is_idempotent_and_releases_recorder_env_and_context():
    recorder = FakeRecorder()
    raw_env = FakeGymEnv()
    wrapped = VideoRecordingWrapper(raw_env, recorder)
    wrapped.close()
    wrapped.close()
    assert recorder.stop_count == 1
    assert raw_env.close_count == 1

    adapter = FakeGymEnvAdapter()
    env = BasicAdroitEnv(adapter, cameras=['top'])
    env.close()
    env.close()
    assert adapter.env.close_count == 1
    assert adapter.env.sim.context.close_count == 1

    vector_env = SimpleNamespace(close_count=0)
    vector_env.close = lambda: setattr(
        vector_env, 'close_count', vector_env.close_count + 1)
    runner = AdroitRunner.__new__(AdroitRunner)
    runner.env = vector_env
    runner._closed = False
    runner.close()
    runner.close()
    assert vector_env.close_count == 1
