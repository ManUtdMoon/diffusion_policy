"""Batched GPU OneLeg adapter. Import isaacgym before torch in live processes."""

import numpy as np
import torch
from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    matrix_to_quaternion
)
from torchvision.transforms import functional as TF


def pack_oneleg_observation(raw_obs):
    """Pack current observations without a time axis; images are CHW in [0, 1]."""
    obs = {}
    for key in ('color_image1', 'color_image2'):
        image = raw_obs[key]
        if image.ndim != 4 or image.shape[-1] != 3 or image.dtype != torch.uint8:
            raise ValueError(f'{key} must be uint8 (B, H, W, 3)')
        image = image.permute(0, 3, 1, 2)
        if key == 'color_image1':
            image = TF.resize(image, [240, 320], antialias=True)
        else:
            height, width = image.shape[-2:]
            resized_width = int(240 * width / height)
            if resized_width < 320:
                raise ValueError('Front camera must be wide enough for a 240x320 crop')
            image = TF.resize(image, [240, resized_width], antialias=True)
            offset = (resized_width - 320) // 2
            image = image[..., offset:offset + 320]
        obs[key] = image.to(dtype=torch.float32) / 255.

    state = raw_obs['robot_state']
    # Isaac Gym xyzw -> PyTorch3D wxyz; rot6d encodes rows without transposing.
    quat = state['ee_quat'].to(dtype=torch.float32)
    rotation = matrix_to_rotation_6d(quaternion_to_matrix(quat[:, [3, 0, 1, 2]]))
    obs['robot_state'] = torch.cat([
        state['ee_pos'], rotation, state['ee_pos_vel'],
        state['ee_ori_vel'], state['gripper_width']], dim=-1).to(dtype=torch.float32)
    batch_size = obs['robot_state'].shape[0]
    if obs['robot_state'].shape != (batch_size, 16):
        raise ValueError('Packed robot_state must be (B, 16)')
    for key in ('color_image1', 'color_image2'):
        if obs[key].shape != (batch_size, 3, 240, 320):
            raise ValueError(f'Invalid batch shape for {key}: {obs[key].shape}')
    return obs


def _make_native_env(render_size, **kwargs):
    import isaacgym
    import gym
    import furniture_bench
    from furniture_bench.sim_config import sim_config

    # Native construction reads this global config; keep the override local.
    camera = sim_config['camera']
    sim_config['camera'] = dict(camera, color_img_size=tuple(render_size))
    try:
        return gym.make('FurnitureSim-v0', **kwargs)
    finally:
        sim_config['camera'] = camera
        gym.logger.set_level(gym.logger.WARN)


class OneLegEnv:
    """Current-frame observations and public 10D rot6d absolute actions.

    Actions are batch-converted to native quaternion format before one step.

    Reward/done retain native (B, 1) shapes and semantics, including the strict
    `env_steps > max_env_steps` timeout and the pre-reward success check in done.
    No action chunks, automatic reset, or episode termination correction is used.
    """

    def __init__(self, max_env_steps=700, num_envs=1, compute_device_id=0,
            graphics_device_id=0, headless=True, randomness='low', render_size=(640, 360)):
        if num_envs < 1 or max_env_steps < 1:
            raise ValueError('num_envs and max_env_steps must be positive')
        if len(render_size) != 2 or any(not isinstance(value, int) or value < 1 for value in render_size):
            raise ValueError('render_size must be positive (width, height) integers')
        self.num_envs = num_envs
        self.env = _make_native_env(
            render_size=render_size, furniture='one_leg', num_envs=num_envs, max_env_steps=max_env_steps,
            compute_device_id=compute_device_id, graphics_device_id=graphics_device_id,
            headless=headless, randomness=randomness, resize_img=False,
            np_step_out=False, channel_first=False, concat_robot_state=False,
            robot_state_as_dict=True, squeeze_batch_dim=False,
            act_rot_repr='quat', action_type='pos', ctrl_mode='osc')
        self.device = self.env.unwrapped.device

    @torch.no_grad()
    def reset(self, seed=None):
        """Reset the full batch; environment i uses seed + i when provided."""
        if seed is None:
            return pack_oneleg_observation(self.env.reset())
        # Native reset handles refresh/bookkeeping; seed each reset_env call.
        native = self.env.unwrapped
        reset_env = native.reset_env
        def seeded_reset(env_index, **kwargs):
            np.random.seed(seed + env_index)
            return reset_env(env_index, **kwargs)
        native.reset_env = seeded_reset
        try:
            return pack_oneleg_observation(self.env.reset())
        finally:
            del native.reset_env

    @torch.no_grad()
    def step(self, action):
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if action.shape != (self.num_envs, 10):
            raise ValueError(f'Action must be ({self.num_envs}, 10), got {action.shape}')
        if not torch.isfinite(action).all():
            raise ValueError('Action must be finite')
        # Preserve native rot6d semantics: clip before converting the whole batch.
        action = action.clamp(-1, 1)
        rotation = matrix_to_quaternion(rotation_6d_to_matrix(action[:, 3:9]))
        native_action = torch.cat([action[:, :3], rotation[:, [1, 2, 3, 0]], action[:, -1:]], dim=-1)
        raw_obs, reward, done, info = self.env.step(native_action)
        return pack_oneleg_observation(raw_obs), reward, done, info

    def is_success(self):
        return torch.tensor([result['task'] for result in self.env.is_success()],
            device=self.device, dtype=torch.bool)

    def close(self):
        if self.env is not None:
            self.env.close()
            # Release the native object so its destructor destroys the simulator.
            self.env = None
