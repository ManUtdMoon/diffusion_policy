import ast
import os
import pathlib
import unittest
from types import SimpleNamespace
from typing import Union
from unittest import mock

import numpy as np
from PIL import Image
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_matrix
from torchvision.transforms import functional as TF, InterpolationMode

from zprl.env.furniturebench.oneleg_env import OneLegEnv, pack_oneleg_observation, _make_native_env


def raw_observation(batch_size=2):
    matrix = axis_angle_to_matrix(torch.tensor([[0.4, -0.7, 1.1]])).expand(batch_size, -1, -1)
    quat = matrix_to_quaternion(matrix)[:, [1, 2, 3, 0]]
    image = (torch.arange(720 * 1280, dtype=torch.int32) % 256).to(torch.uint8)
    image = image.reshape(1, 720, 1280, 1).expand(batch_size, -1, -1, 3)
    return {
        'color_image1': image,
        'color_image2': image.flip(2),
        'robot_state': {
            'ee_pos': torch.full((batch_size, 3), 0.1),
            'ee_quat': quat,
            'ee_pos_vel': torch.full((batch_size, 3), 0.2),
            'ee_ori_vel': torch.full((batch_size, 3), 0.3),
            'gripper_width': torch.full((batch_size, 1), 0.04),
            'joint_positions': torch.ones(batch_size, 7),
        },
        'parts_poses': torch.zeros(batch_size, 35),
        'success': torch.ones(batch_size),
    }


class FakeNativeEnv:
    device = torch.device('cpu')

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.unwrapped = self
        self.raw = raw_observation(kwargs['num_envs'])
        self.reset_count = 0
        self.actions = []
        self.close_count = 0
        self.reward = torch.tensor([[1.], [0.]])
        self.done = torch.tensor([[False], [True]])

    def reset(self):
        self.reset_count += 1
        self.seed_probes = []
        for i in range(self.kwargs['num_envs']):
            self.reset_env(i)
        self.seed_probe = self.seed_probes[0]
        return self.raw

    def reset_env(self, env_index):
        self.seed_probes.append(np.random.uniform())

    def step(self, action):
        self.actions.append(action.clone())
        return self.raw, self.reward, self.done, {'obs_success': True}

    def is_success(self):
        return [{'task': True}, {'task': False}]

    def close(self):
        self.close_count += 1


class OneLegEnvTest(unittest.TestCase):
    def test_observation_shape_scaling_state_order_and_rotation_rows(self):
        raw = raw_observation()
        obs = pack_oneleg_observation(raw)
        self.assertEqual(set(obs), {'color_image1', 'color_image2', 'robot_state'})
        for key in ('color_image1', 'color_image2'):
            self.assertEqual(tuple(obs[key].shape), (2, 3, 240, 320))
            self.assertEqual(obs[key].dtype, torch.float32)
            self.assertGreaterEqual(obs[key].amin().item(), 0)
            self.assertLessEqual(obs[key].amax().item(), 1)
        expected_rotation = axis_angle_to_matrix(torch.tensor([[0.4, -0.7, 1.1]]))
        torch.testing.assert_close(obs['robot_state'][:, :3], raw['robot_state']['ee_pos'])
        torch.testing.assert_close(obs['robot_state'][:, 3:9],
            expected_rotation[:, :2, :].reshape(1, 6).expand(2, -1))
        torch.testing.assert_close(obs['robot_state'][:, 9:12], raw['robot_state']['ee_pos_vel'])
        torch.testing.assert_close(obs['robot_state'][:, 12:15], raw['robot_state']['ee_ori_vel'])
        torch.testing.assert_close(obs['robot_state'][:, 15:], raw['robot_state']['gripper_width'])
        wrist = TF.resize(raw['color_image1'].permute(0, 3, 1, 2), [240, 320], antialias=True)
        front = TF.resize(raw['color_image2'].permute(0, 3, 1, 2), [240, 426], antialias=True)
        torch.testing.assert_close(obs['color_image1'], wrist.float() / 255.)
        torch.testing.assert_close(obs['color_image2'], front[..., 53:373].float() / 255.)

    @mock.patch('zprl.env.furniturebench.oneleg_env._make_native_env', side_effect=FakeNativeEnv)
    def test_native_configuration_one_step_done_and_explicit_reset(self, factory):
        env = OneLegEnv(max_env_steps=9, num_envs=2)
        native = env.env
        self.assertEqual(native.kwargs['furniture'], 'one_leg')
        self.assertEqual(native.kwargs['act_rot_repr'], 'quat')
        self.assertEqual(native.kwargs['action_type'], 'delta')
        self.assertFalse(native.kwargs['resize_img'])
        self.assertEqual(native.kwargs['render_size'], (640, 360))
        env.reset(seed=17)
        probe = native.seed_probe
        self.assertEqual(native.seed_probes,
            [np.random.RandomState(seed).uniform() for seed in (17, 18)])
        self.assertEqual(native.reset_env.__func__, FakeNativeEnv.reset_env)
        self.assertNotIn('reset_env', vars(native))
        env.reset(seed=17)
        self.assertEqual(probe, native.seed_probe)
        action = np.zeros((2, 10), dtype=np.float64)
        # Public rot6d actions become native xyzw quaternions without rescaling
        # translation or binarizing the gripper.
        action[:, :3] = 0.04
        action[:, 3:9] = [1, 0, 0, 0, 1, 0]
        action[:, -1] = -0.25
        obs, reward, done, info = env.step(action)
        self.assertEqual(len(native.actions), 1)
        expected = torch.tensor([[0.04, 0.04, 0.04, 0, 0, 0, 1, -0.25]]).repeat(2, 1)
        torch.testing.assert_close(native.actions[0], expected)
        self.assertIs(reward, native.reward)
        self.assertIs(done, native.done)
        self.assertEqual(native.reset_count, 2)
        self.assertEqual(info, {'obs_success': True})
        torch.testing.assert_close(env.is_success(), torch.tensor([True, False]))
        for bad in (torch.zeros(10), torch.zeros(2, 16, 10), torch.full((2, 10), float('nan'))):
            with self.assertRaises(ValueError):
                env.step(bad)
        self.assertEqual(len(native.actions), 1)
        env.close()
        env.close()
        self.assertEqual(native.close_count, 1)

    @mock.patch('zprl.env.furniturebench.oneleg_env._make_native_env', side_effect=FakeNativeEnv)
    def test_batch_conversion_clips_rot6d_before_normalization(self, factory):
        env = OneLegEnv(num_envs=2)
        action = torch.tensor([[2., -2., .02, 2., .6, -.2, .4, 3., -.8, -2.],
            [-.01, .02, -.03, 1., 0., 0., 0., 1., 0., .25]])
        env.step(action)
        native_action = env.env.actions[0]
        self.assertEqual(native_action.shape, (2, 8))
        torch.testing.assert_close(native_action[:, :3], action[:, :3].clamp(-1, 1))
        torch.testing.assert_close(native_action[:, -1], action[:, -1].clamp(-1, 1))
        # Native rot6d constructs these orthogonal rows after component clipping.
        a = action[:, 3:6].clamp(-1, 1)
        b = action[:, 6:9].clamp(-1, 1)
        first = torch.nn.functional.normalize(a, dim=-1)
        second = torch.nn.functional.normalize(b - (first*b).sum(-1, keepdim=True)*first, dim=-1)
        expected = torch.stack([first, second, torch.cross(first, second, dim=-1)], dim=-2)
        actual = quaternion_to_matrix(native_action[:, [6,3,4,5]])
        torch.testing.assert_close(actual, expected)

    def test_camera_override_is_used_during_construction_and_always_restored(self):
        camera = {'color_img_size': (1280, 720), 'resized_img_size': (224, 224)}
        sim_config = {'camera': camera}
        modules = {'isaacgym': SimpleNamespace(), 'furniture_bench': SimpleNamespace(),
            'furniture_bench.sim_config': SimpleNamespace(sim_config=sim_config)}
        def make(*args, **kwargs):
            self.assertEqual(args[0], 'FurnitureSim-v0')
            self.assertEqual(sim_config['camera']['color_img_size'], (640, 360))
            self.assertEqual(camera['color_img_size'], (1280, 720))
            return object()
        for fail in (False, True):
            with self.subTest(fail=fail):
                def construct(*args, **kwargs):
                    result = make(*args, **kwargs)
                    if fail:
                        raise RuntimeError('camera creation failed')
                    return result
                with mock.patch.dict('sys.modules', dict(modules, gym=SimpleNamespace(make=construct))):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'camera creation failed'):
                            _make_native_env((640, 360), furniture='one_leg')
                    else:
                        _make_native_env((640, 360), furniture='one_leg')
                self.assertIs(sim_config['camera'], camera)

    def test_invalid_image_contract_is_rejected(self):
        raw = raw_observation()
        raw['color_image1'] = raw['color_image1'].float()
        with self.assertRaises(ValueError):
            pack_oneleg_observation(raw)

    @unittest.skipUnless(os.environ.get('ONELEG_SIM_REPO'),
        'set ONELEG_SIM_REPO to compare the author image preprocessing')
    def test_author_image_preprocessing_matches(self):
        path = pathlib.Path(os.environ['ONELEG_SIM_REPO']) / 'src/data_processing/utils.py'
        module = ast.parse(path.read_text())
        module.body = [node for node in module.body if isinstance(node, ast.FunctionDef)
                       and node.name in ('resize', 'resize_crop')]
        namespace = {'np': np, 'torch': torch, 'Union': Union, 'Image': Image,
                     'F': TF, 'InterpolationMode': InterpolationMode}
        exec(compile(module, str(path), 'exec'), namespace)
        raw = raw_observation()
        obs = pack_oneleg_observation(raw)
        for key, name in (('color_image1', 'resize'), ('color_image2', 'resize_crop')):
            expected = namespace[name](raw[key]).permute(0, 3, 1, 2).float() / 255.
            torch.testing.assert_close(obs[key], expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
