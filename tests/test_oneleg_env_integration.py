import os
import pathlib
import subprocess
import sys
import unittest


LIVE_CHECK = r'''
import isaacgym
import gc
import os
import sys
import weakref
import numpy as np
import torch
from zprl.env.furniturebench.oneleg_env import OneLegEnv

count = int(sys.argv[1])
render_size = tuple(map(int, sys.argv[2:4]))
from furniture_bench.sim_config import sim_config
camera = sim_config['camera']
env = OneLegEnv(num_envs=count, max_env_steps=3, render_size=render_size)
assert sim_config['camera'] is camera
assert env.env.unwrapped.img_size == render_size
assert env.env.unwrapped.act_rot_repr == 'quat'
assert set(env.env.unwrapped.camera_handles) == {'wrist', 'front'}
assert 'parts_poses' not in env.env.unwrapped.obs_keys
assert not any(key.startswith('depth') for key in env.env.unwrapped.obs_keys)
assert np.isclose(env.env.unwrapped.camera_cfg.horizontal_fov, 69.4)
raw = env.env.get_observation()
assert raw['color_image1'].shape == (count, render_size[1], render_size[0], 3)
del raw
native_ref = weakref.ref(env.env.unwrapped)
try:
    def reset_orientations():
        return np.stack([np.concatenate([part.reset_ori[0].reshape(-1)
            for part in furniture.parts]) for furniture in env.env.unwrapped.furnitures])
    obs = env.reset(seed=7)
    initial_orientations = reset_orientations()
    initial_root_poses = env.env.unwrapped.root_tensor[..., :7].clone()
    for key in ('color_image1', 'color_image2'):
        assert obs[key].shape == (count, 3, 240, 320)
        assert obs[key].device == torch.device('cuda:0')
        assert obs[key].dtype == torch.float32
        assert 0 <= obs[key].min() <= obs[key].max() <= 1
        assert obs[key].std() > 0
    assert obs['robot_state'].shape == (count, 16)
    assert torch.isfinite(obs['robot_state']).all()
    start = env.env.unwrapped.isaac_gym.get_sim_time(env.env.unwrapped.sim)
    neutral = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 1, 0, -1.]], device=env.device).repeat(count, 1)
    for step in range(3):
        obs, reward, done, info = env.step(neutral)
        assert reward.shape == done.shape == (count, 1)
        assert not done.any()
    obs, reward, done, info = env.step(neutral)
    assert done.all()
    assert env.is_success().shape == (count,)
    elapsed = env.env.unwrapped.isaac_gym.get_sim_time(env.env.unwrapped.sim) - start
    assert np.isclose(elapsed, 0.2), elapsed
    assert torch.all(env.env.unwrapped.env_steps == 4)
    obs = env.reset(seed=8)
    assert torch.all(env.env.unwrapped.env_steps == 0)
    assert not env.is_success().any()
    if count == 2:
        np.testing.assert_allclose(reset_orientations()[0], initial_orientations[1], rtol=0, atol=1e-7)
    obs = env.reset(seed=7)
    np.testing.assert_allclose(reset_orientations(), initial_orientations, rtol=0, atol=1e-7)
    torch.testing.assert_close(env.env.unwrapped.root_tensor[..., :7],
        initial_root_poses, rtol=0, atol=2e-4)
    assert 'reset_env' not in vars(env.env.unwrapped)
    checkpoint = os.environ.get('ONELEG_ENV_CHECKPOINT')
    if checkpoint:
        import dill
        import hydra
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        payload = torch.load(checkpoint, map_location='cpu', pickle_module=dill)
        policy = hydra.utils.instantiate(payload['cfg'].policy)
        policy.load_state_dict(payload['state_dicts']['ema_model'])
        policy = policy.to(env.device).requires_grad_(False).eval()
        with torch.no_grad():
            result = policy.predict_action({key: value[:, None] for key, value in obs.items()})
        assert result['action_pred'].shape == (count, 16, 10)
        assert torch.isfinite(result['action_pred']).all()
        obs, reward, done, info = env.step(result['action'][:, 0])
        assert torch.isfinite(obs['robot_state']).all()
        print('EMA_LIVE_STEP_OK', count, flush=True)
    print('NATIVE_OK', count, 'four_steps_seconds', elapsed, flush=True)
finally:
    env.close()
    env.close()
    gc.collect()
assert native_ref() is None, 'Native simulator still retained after close'
print('CLOSE_OK', count, flush=True)
'''


@unittest.skipUnless(os.environ.get('ONELEG_ENV_GPU_SMOKE') == '1',
    'Opt in to live Isaac Gym tests, imported before torch in child processes')
class OneLegEnvIntegrationTest(unittest.TestCase):
    def test_single_and_two_native_environments(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='1,0,2,3')
        for count, render_size in ((1, (1280, 720)), (2, (640, 360))):
            with self.subTest(num_envs=count, render_size=render_size):
                result = subprocess.run([sys.executable, '-c', LIVE_CHECK, str(count), *map(str, render_size)],
                    cwd=root, env=environment, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn(f'NATIVE_OK {count}', result.stdout)
                self.assertIn(f'CLOSE_OK {count}', result.stdout)
                if environment.get('ONELEG_ENV_CHECKPOINT'):
                    self.assertIn(f'EMA_LIVE_STEP_OK {count}', result.stdout)


if __name__ == '__main__':
    unittest.main()
