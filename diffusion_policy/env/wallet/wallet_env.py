import time
from collections import deque

import cv2
import numpy as np
from gym import spaces
from pynput import keyboard
from scipy.spatial.transform import Rotation as R, Slerp

from diffusion_policy.env.box.franka_wrapper import FrankaWrapper
from diffusion_policy.env.box.xarm_wrapper import XArmWrapper
from diffusion_policy.env.flip.franka.common.precise_sleep import precise_wait
from diffusion_policy.env.wallet.realsense_multiview import (
    DEFAULT_MULTI_CAMERAS,
    MultiViewRealSense,
)
from diffusion_policy.env.wallet.robotiq_wrapper import (
    GRIPPER_OPEN_POSITION,
    RobotiqWrapper,
)


XARM_MIN_Z_MM = 258.0  # change to real tcp, before: 328.0 mm, delta = -70.0 mm

is_done = False


def on_press(key):
    global is_done
    try:
        if hasattr(key, "char") and key.char:
            ch = key.char
            if ch.isdigit() or ch.isalpha() or ch in ["+", "-"]:
                is_done = True
        elif hasattr(key, "vk") and key.vk == 65437:
            is_done = True
    except AttributeError:
        pass


keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()


class WalletEnv:
    """
    Env for deploying policy on the wallet task.

    Action layout:
    [xarm_pos(3,m), xarm_quat(4), xarm_gripper(1),
     franka_pos(3,m), franka_quat(4), franka_gripper(1)]

    Observation:
    - qpos: (16,) = [xarm_q(6), xarm_gripper(1), franka_q(7), franka_gripper(1)]
    - images: dict(cam_name -> (3,120,160), float32 in [0,1], RGB)
    """

    def __init__(
        self,
        dt=1 / 20.0,
        camera_cfg=None,
        smooth=False,
        smooth_weight=0.7,
        smooth_steps=3,
    ):
        self.dt = dt
        self.camera_cfg = camera_cfg if camera_cfg is not None else DEFAULT_MULTI_CAMERAS
        self.smooth = bool(smooth)
        self.smooth_weight = float(smooth_weight)
        self.smooth_steps = int(smooth_steps)
        if self.smooth_steps < 1:
            raise ValueError(f"smooth_steps must be >= 1, got {self.smooth_steps}")
        if not (0.0 <= self.smooth_weight <= 1.0):
            raise ValueError(f"smooth_weight must be in [0, 1], got {self.smooth_weight}")
        self._xarm_pose_buffer = deque(maxlen=self.smooth_steps)
        self._franka_pose_buffer = deque(maxlen=self.smooth_steps)

        self.franka = FrankaWrapper(joints_init=(-0.0465, -0.7345, 0.5555, -2.6251, 1.0675, 2.0182, -1.5647))
        self.franka_gripper = RobotiqWrapper(robot="franka")
        time.sleep(1.0)
        self.xarm = XArmWrapper(joints_init=[-50.5, 9.2, -99.0, -0.3, 89.2, -108.4])
        self.xarm_gripper = RobotiqWrapper(robot="xarm")
        self.xarm_gripper.open()
        self.franka_gripper.open()

        self.camera = MultiViewRealSense(self.camera_cfg)
        self.camera.start()

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32),
                "global": spaces.Box(low=0.0, high=1.0, shape=(3, 120, 160), dtype=np.float32),
                "wrist_0": spaces.Box(low=0.0, high=1.0, shape=(3, 120, 160), dtype=np.float32),
                "wrist_1": spaces.Box(low=0.0, high=1.0, shape=(3, 120, 160), dtype=np.float32),
            }
        )

        print("WalletEnv initialized.")

    def _get_obs(self):
        xarm_q = self.xarm.get_joint()
        xarm_gripper_state = self.xarm_gripper.get_state()
        franka_q = self.franka.get_joint()
        franka_gripper_state = self.franka_gripper.get_state()
        qpos = np.concatenate(
            [xarm_q, [xarm_gripper_state], franka_q, [franka_gripper_state]]
        ).astype(np.float32)

        obs = {"qpos": qpos}
        frames = self.camera.get_frames()
        for cam_name, frame in frames.items():
            obs[cam_name] = self._transform_image(frame, cam_name)

        return obs

    def _apply_action(self, action14):
        # xArm
        xarm_target = np.array(action14[:6], dtype=np.float32)
        xarm_cmd = xarm_target.copy()
        xarm_cmd[:3] *= 1000.0
        xarm_cmd[2] = max(xarm_cmd[2], XARM_MIN_Z_MM)
        xarm_cmd[3:] = xarm_cmd[3:] * 180.0 / np.pi
        self.xarm.set_servo_cartesian(xarm_cmd)
        if action14[6] > 0.5:
            self.xarm_gripper.close()
        else:
            self.xarm_gripper.open()

        # Franka
        franka_target = np.array(action14[7:13], dtype=np.float32)
        self.franka.franka.servoL(franka_target, self.dt)
        franka_gripper_target = int(
            GRIPPER_OPEN_POSITION
            + (1.0 - float(action14[13])) * (255 - GRIPPER_OPEN_POSITION)
        )
        self.franka_gripper.set_position(franka_gripper_target)

    def reset(self):
        print("<RESET>")
        global is_done
        self.env_step = 0
        self.done = False
        is_done = False
        self._xarm_pose_buffer.clear()
        self._franka_pose_buffer.clear()

        # reset robot
        self.franka.reset()
        time.sleep(1.0)
        self.xarm.reset()
        time.sleep(1.0)
        # self.franka_gripper.open()
        # self.xarm_gripper.open()
        input("Press Enter to continue...")

        obs = self._get_obs()

        self.t_start = time.monotonic()
        return obs

    def reset_end(self):
        self.xarm_gripper.open()
        self.franka_gripper.open()

    def step(self, action):
        global is_done
        self.env_step += 1
        t_cycle_end = self.t_start + self.env_step * self.dt

        if not self.done:
            # control the robot
            action14 = self._transform_action(action)
            if self.smooth:
                action14 = self._smooth_action14(action14)
            self._apply_action(action14)

        precise_wait(t_cycle_end)

        obs = self._get_obs()

        self.done, timeout = self.terminate(is_done)

        # post-process
        reward = 0.0
        is_success = False
        if self.done:
            user_input = input("\nEpisode ended. [0-9]=success, [a-z]=failure.").strip()
            if user_input:
                label = user_input[-1]

            is_success = label.isdigit()
            reward = 1.0 if is_success else 0.0
            print(f"{'Success' if is_success else 'Failure'} recorded!")

        return obs, reward, self.done, {"is_success": is_success, "timeout": timeout}

    def terminate(self, is_done):
        '''
        return done, timeout
        '''
        if is_done:
            return True, False
        if self.env_step >= 1000:
            return True, True
        return False, False

    def close(self):
        try:
            self.xarm_gripper.open()
            self.franka_gripper.open()
        except Exception:
            pass
        try:
            self.xarm_gripper.shutdown()
            self.franka_gripper.shutdown()
        except Exception:
            pass
        try:
            self.xarm.close()
        except Exception:
            pass
        try:
            self.franka.close()
        except Exception:
            pass
        try:
            self.camera.stop()
        except Exception:
            pass

    def _transform_image(self, raw_img, cam_name):
        rgb = raw_img[..., ::-1]
        if cam_name == "global":
            rgb = rgb[0:150, 70:270]
        resized_img = cv2.resize(rgb, (160, 120), interpolation=cv2.INTER_AREA)
        return resized_img.transpose(2, 0, 1).astype(np.float32) / 255.0

    def _transform_action(self, action):
        """
        Input:
        - (14,): [xarm_pos3, xarm_rpy3, xarm_g1, franka_pos3, franka_rotvec3, franka_g1]
        - (16,): [xarm_pos3, xarm_quat4, xarm_g1, franka_pos3, franka_quat4, franka_g1]
        Output:
        - (14,) with xarm_rpy + franka_rotvec
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] == 14:
            return action
        if action.shape[0] != 16:
            raise ValueError(f"Expected action shape (14,) or (16,), got {action.shape}")

        xarm_quat = action[3:7].astype(np.float64)
        franka_quat = action[11:15].astype(np.float64)
        xarm_quat /= max(np.linalg.norm(xarm_quat), 1e-12)
        franka_quat /= max(np.linalg.norm(franka_quat), 1e-12)

        xarm_rpy = R.from_quat(xarm_quat).as_euler("xyz", degrees=False).astype(np.float32)
        franka_rotvec = R.from_quat(franka_quat).as_rotvec().astype(np.float32)

        return np.concatenate(
            [
                action[0:3],
                xarm_rpy,
                action[7:8],
                action[8:11],
                franka_rotvec,
                action[15:16],
            ],
            axis=0,
        ).astype(np.float32)

    def _smooth_action14(self, action14):
        action14 = np.asarray(action14, dtype=np.float64).copy()

        self._xarm_pose_buffer.append(action14[:6].copy())
        if len(self._xarm_pose_buffer) >= 2:
            action14[:6] = self._compute_smoothed_pose(
                self._xarm_pose_buffer,
                rotation_repr="euler_xyz",
            )

        self._franka_pose_buffer.append(action14[7:13].copy())
        if len(self._franka_pose_buffer) >= 2:
            action14[7:13] = self._compute_smoothed_pose(
                self._franka_pose_buffer,
                rotation_repr="rotvec",
            )

        return action14.astype(np.float32)

    def _compute_smoothed_pose(self, pose_buffer, rotation_repr):
        """
        Smooth pose in [pos(3), rot(3)] with exponential weighting + sequential SLERP.
        rotation_repr: "euler_xyz" or "rotvec"
        """
        if len(pose_buffer) < 2:
            return pose_buffer[-1].copy()

        poses = list(pose_buffer)[::-1]
        positions = np.array([p[:3] for p in poses], dtype=np.float64)

        if rotation_repr == "euler_xyz":
            rotations = [R.from_euler("xyz", p[3:], degrees=False) for p in poses]
        elif rotation_repr == "rotvec":
            rotations = [R.from_rotvec(p[3:]) for p in poses]
        else:
            raise ValueError(f"Unsupported rotation_repr: {rotation_repr}")

        weights = np.array([self.smooth_weight ** i for i in range(len(poses))], dtype=np.float64)
        weights = weights / max(weights.sum(), 1e-12)
        smoothed_pos = np.average(positions, axis=0, weights=weights)

        smoothed_rot = rotations[0]
        for i in range(1, len(rotations)):
            blend_weight = weights[i] / max(weights[: i + 1].sum(), 1e-12)
            slerp = Slerp([0.0, 1.0], R.concatenate([smoothed_rot, rotations[i]]))
            smoothed_rot = slerp([blend_weight])[0]

        smoothed_pose = np.empty(6, dtype=np.float64)
        smoothed_pose[:3] = smoothed_pos
        if rotation_repr == "euler_xyz":
            smoothed_pose[3:] = smoothed_rot.as_euler("xyz", degrees=False)
        else:
            smoothed_pose[3:] = smoothed_rot.as_rotvec()
        return smoothed_pose
