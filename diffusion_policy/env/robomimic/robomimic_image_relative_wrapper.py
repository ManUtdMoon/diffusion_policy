from typing import Dict

import gym
from gym import spaces
import numpy as np

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pose_util import mat_to_pose, mat_to_pose10d, pose10d_to_mat
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


def _resolve_raw_obs_key(key: str) -> str:
    if key.endswith('_rot6d'):
        return key[:-len('_rot6d')] + '_quat'
    return key


def _build_pose_mat(pos: np.ndarray, rot_mat: np.ndarray) -> np.ndarray:
    mats = np.zeros(pos.shape[:-1] + (4, 4), dtype=np.float32)
    mats[..., :3, :3] = rot_mat.astype(np.float32)
    mats[..., :3, 3] = pos.astype(np.float32)
    mats[..., 3, 3] = 1.0
    return mats


class RobomimicImageRelativeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, shape_meta: dict):
        super().__init__(env)
        self.shape_meta = shape_meta
        self.rotation_transformer = RotationTransformer('quaternion', 'matrix')
        self.raw_action_rotation_transformer = RotationTransformer('rotation_6d', 'matrix')
        self.raw_obs = None

        self.pose_robot_prefixes = list()
        for key in self.shape_meta['obs'].keys():
            if key.endswith('_eef_pos'):
                prefix = key[:-len('_eef_pos')]
                if f'{prefix}_eef_rot6d' in self.shape_meta['obs']:
                    self.pose_robot_prefixes.append(prefix)
        if len(self.pose_robot_prefixes) != 1:
            raise RuntimeError('relative robomimic image wrapper currently supports single-arm only')

        self.observation_space = spaces.Dict()
        for key, value in self.shape_meta['obs'].items():
            shape = value['shape']
            type = value.get('type', 'low_dim')
            if type == 'rgb':
                self.observation_space[key] = spaces.Box(
                    low=0,
                    high=1,
                    shape=env.observation_space[_resolve_raw_obs_key(key)].shape,
                    dtype=np.float32,
                )
            else:
                raw_key = _resolve_raw_obs_key(key)
                raw_shape = env.observation_space[raw_key].shape
                seq_shape = raw_shape[:-1]
                if key.endswith('_eef_pos') or key.endswith('_gripper_qpos'):
                    low, high = -1, 1
                elif key.endswith('_eef_rot6d'):
                    low, high = -1, 1
                else:
                    raise RuntimeError(f'Unsupported relative obs key {key}')
                self.observation_space[key] = spaces.Box(
                    low=low,
                    high=high,
                    shape=seq_shape + tuple(shape),
                    dtype=np.float32,
                )

        action_shape = env.action_space.shape[:-1] + tuple(self.shape_meta['action']['shape'])
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32,
        )

    def _transform_obs(self, raw_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        obs_dict = dict()
        anchor_pose_mat_dict = dict()
        for key in self.shape_meta['obs'].keys():
            if self.shape_meta['obs'][key].get('type', 'low_dim') == 'rgb':
                obs_dict[key] = raw_obs[_resolve_raw_obs_key(key)].astype(np.float32)

        for prefix in self.pose_robot_prefixes:
            pos_key = f'{prefix}_eef_pos'
            quat_key = f'{prefix}_eef_quat'
            rot_key = f'{prefix}_eef_rot6d'

            pos = raw_obs[pos_key].astype(np.float32)
            quat = raw_obs[quat_key].astype(np.float32)
            rot_mat = self.rotation_transformer.forward(quat)
            pose_mat = _build_pose_mat(pos, rot_mat)
            anchor_pose_mat = pose_mat[-1].copy()
            anchor_pose_mat_dict[prefix] = anchor_pose_mat
            rel_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=anchor_pose_mat,
                pose_rep='relative',
                backward=False,
            )
            rel_pose = mat_to_pose10d(rel_pose_mat)
            obs_dict[pos_key] = rel_pose[:, :3].astype(np.float32)
            obs_dict[rot_key] = rel_pose[:, 3:].astype(np.float32)

        for key in self.shape_meta['obs'].keys():
            if key in obs_dict:
                continue
            raw_key = _resolve_raw_obs_key(key)
            obs_dict[key] = raw_obs[raw_key].astype(np.float32)
        return obs_dict

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        anchor_prefix = self.pose_robot_prefixes[0]
        anchor_pose_mat = _build_pose_mat(
            self.raw_obs[f'{anchor_prefix}_eef_pos'].astype(np.float32),
            self.rotation_transformer.forward(self.raw_obs[f'{anchor_prefix}_eef_quat'].astype(np.float32))
        )[-1].copy()

        action_pose = action[..., :9].astype(np.float32)
        action_gripper = action[..., 9:].astype(np.float32)
        rel_action_mat = pose10d_to_mat(action_pose)
        abs_action_mat = convert_pose_mat_rep(
            rel_action_mat,
            base_pose_mat=anchor_pose_mat,
            pose_rep='relative',
            backward=True,
        )
        abs_action_pose = mat_to_pose(abs_action_mat).astype(np.float32)
        return np.concatenate([abs_action_pose, action_gripper], axis=-1).astype(np.float32)

    def reset(self):
        raw_obs = self.env.reset()
        self.raw_obs = raw_obs
        return self._transform_obs(raw_obs)

    def step(self, action):
        raw_action = self._decode_action(action)
        raw_obs, reward, done, info = self.env.step(raw_action)
        self.raw_obs = raw_obs
        obs = self._transform_obs(raw_obs)
        return obs, reward, done, info
