import json
import os
import time
from collections import deque
from pathlib import Path

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


XARM_MIN_Z_MM = 256.0  # change to real tcp, before: 328.0 mm, delta = -70.0 mm
XARM_MIN_QUAT_NORM = 0.5

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
        trace_log_path=None,
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
        self._last_valid_xarm_pose = None
        self._last_xarm_action_fallback = None
        self._xarm_quat_fallback_count = 0
        self._xarm_quat_fallback_first_step = None
        self._xarm_quat_min_norm = None
        self._last_trace = None
        self._trace_path = self._get_trace_path(trace_log_path)
        self._trace_file = None

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
        if self._trace_path is not None:
            print(f"WalletEnv trace log: {self._trace_path}")

    def _get_trace_path(self, trace_log_path):
        if os.environ.get("WALLET_ENV_TRACE", "1") == "0":
            return None
        if trace_log_path is None:
            trace_dir = Path(os.environ.get("WALLET_ENV_TRACE_DIR", "data/debug/wallet_env"))
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_log_path = trace_dir / "wallet_env_trace_latest.jsonl"
        else:
            trace_log_path = Path(trace_log_path)
            trace_log_path.parent.mkdir(parents=True, exist_ok=True)
        return trace_log_path

    def _reset_trace_file(self):
        if self._trace_path is None:
            return
        if self._trace_file is not None:
            self._trace_file.close()
        self._trace_file = self._trace_path.open("w", encoding="utf-8", buffering=1)

    def _close_trace_file(self):
        if self._trace_file is None:
            return
        self._trace_file.flush()
        self._trace_file.close()
        self._trace_file = None

    def _jsonify(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {k: self._jsonify(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonify(v) for v in value]
        return value

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
        return {
            "xarm_target_m_rad": xarm_target,
            "xarm_cmd_mm_deg": xarm_cmd,
            "xarm_gripper_close": bool(action14[6] > 0.5),
            "franka_target_m_rotvec": franka_target,
            "franka_gripper_target": franka_gripper_target,
        }

    def reset(self):
        print("<RESET>")
        global is_done
        self.env_step = 0
        self.done = False
        is_done = False
        self._xarm_pose_buffer.clear()
        self._franka_pose_buffer.clear()
        self._last_valid_xarm_pose = None
        self._last_xarm_action_fallback = None
        self._xarm_quat_fallback_count = 0
        self._xarm_quat_fallback_first_step = None
        self._xarm_quat_min_norm = None
        self._last_trace = None
        self._reset_trace_file()

        # reset robot
        self.franka.reset()
        time.sleep(1.0)
        self.xarm.reset()
        time.sleep(1.0)
        # self.franka_gripper.open()
        # self.xarm_gripper.open()
        try:
            self._last_valid_xarm_pose = np.asarray(self.xarm.get_position(), dtype=np.float32)
        except Exception as exc:
            print(f"Failed to initialize xarm fallback pose: {exc}")
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
            raw_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
            action14_before_smooth = self._transform_action(raw_action)
            action14 = action14_before_smooth.copy()
            if self.smooth:
                action14 = self._smooth_action14(action14)
            command_info = self._apply_action(action14)
        else:
            raw_action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
            action14_before_smooth = None
            action14 = None
            command_info = None

        precise_wait(t_cycle_end)

        obs = self._get_obs()
        self._write_trace(
            raw_action=raw_action,
            action14_before_smooth=action14_before_smooth,
            action14=action14,
            command_info=command_info,
            obs=obs,
        )

        self.done, timeout = self.terminate(is_done)

        # post-process
        reward = 0.0
        is_success = False
        label = None
        if self.done:
            user_input = input(
                "\nEpisode ended. [0-9]=success, [a-z]=failure, +/-=negative."
            ).strip()
            if user_input:
                label = user_input[-1]

            if label in ["+", "-"]:
                is_success = False
                reward = -0.5
                print("Negative failure recorded!")
            else:
                is_success = bool(label is not None and label.isdigit())
                reward = 1.0 if is_success else 0.0
                print(f"{'Success' if is_success else 'Failure'} recorded!")
            self._print_xarm_quat_summary()
            self._close_trace_file()

        return obs, reward, self.done, {
            "is_success": is_success,
            "timeout": timeout,
            "label": label,
        }

    def _print_xarm_quat_summary(self):
        if self._xarm_quat_fallback_count == 0:
            print("XArm quat abnormal: no")
        else:
            print(
                "XArm quat abnormal: yes, "
                f"count={self._xarm_quat_fallback_count}, "
                f"first_step={self._xarm_quat_fallback_first_step}, "
                f"min_norm={self._xarm_quat_min_norm:.4f}"
            )

    def _write_trace(self, raw_action, action14_before_smooth, action14, command_info, obs):
        if self._trace_file is None:
            return

        qpos = obs.get("qpos", None)
        try:
            xarm_tcp_pose_m_rad = self.xarm.get_position()
        except Exception as exc:
            xarm_tcp_pose_m_rad = {"error": repr(exc)}
        xarm_quat = None
        xarm_quat_norm = None
        xarm_quat_delta_deg = None
        if raw_action.shape[0] == 16:
            xarm_quat = raw_action[3:7].astype(np.float64)
            xarm_quat_norm = float(np.linalg.norm(xarm_quat))

        prev = self._last_trace
        xarm_cmd_delta = None
        xarm_rpy_delta_rad = None
        xarm_rpy_delta_unwrapped_rad = None
        if prev is not None and command_info is not None:
            xarm_cmd_delta = command_info["xarm_cmd_mm_deg"] - prev["xarm_cmd_mm_deg"]
            xarm_rpy_delta_rad = action14[3:6] - prev["xarm_action14"][3:6]
            xarm_rpy_delta_unwrapped_rad = (
                (xarm_rpy_delta_rad + np.pi) % (2.0 * np.pi)
            ) - np.pi
            if xarm_quat is not None and prev["xarm_quat"] is not None:
                q_curr = xarm_quat / max(np.linalg.norm(xarm_quat), 1e-12)
                q_prev = prev["xarm_quat"] / max(np.linalg.norm(prev["xarm_quat"]), 1e-12)
                dot = abs(float(np.dot(q_curr, q_prev)))
                dot = min(1.0, max(-1.0, dot))
                xarm_quat_delta_deg = float(2.0 * np.arccos(dot) * 180.0 / np.pi)

        record = {
            "wall_time": time.time(),
            "env_step": self.env_step,
            "dt": self.dt,
            "smooth": self.smooth,
            "raw_action": raw_action,
            "action14_before_smooth": action14_before_smooth,
            "action14": action14,
            "command": command_info,
            "xarm_quat_norm": xarm_quat_norm,
            "xarm_quat_delta_deg": xarm_quat_delta_deg,
            "xarm_action_fallback": self._last_xarm_action_fallback,
            "xarm_cmd_delta_mm_deg": xarm_cmd_delta,
            "xarm_rpy_delta_rad": xarm_rpy_delta_rad,
            "xarm_rpy_delta_unwrapped_rad": xarm_rpy_delta_unwrapped_rad,
            "qpos": qpos,
            "xarm_qpos": None if qpos is None else qpos[:6],
            "xarm_tcp_pose_m_rad": xarm_tcp_pose_m_rad,
            "xarm_gripper_state": None if qpos is None else qpos[6],
            "franka_qpos": None if qpos is None else qpos[7:14],
            "franka_gripper_state": None if qpos is None else qpos[15],
        }
        self._trace_file.write(json.dumps(self._jsonify(record), ensure_ascii=True) + "\n")

        if command_info is not None:
            self._last_trace = {
                "xarm_cmd_mm_deg": command_info["xarm_cmd_mm_deg"].copy(),
                "xarm_action14": action14.copy(),
                "xarm_quat": None if xarm_quat is None else xarm_quat.copy(),
            }

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
        try:
            self._close_trace_file()
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
            self._last_xarm_action_fallback = None
            return action
        if action.shape[0] != 16:
            raise ValueError(f"Expected action shape (14,) or (16,), got {action.shape}")

        xarm_quat = action[3:7].astype(np.float64)
        franka_quat = action[11:15].astype(np.float64)
        xarm_quat_norm = np.linalg.norm(xarm_quat)
        if self._xarm_quat_min_norm is None:
            self._xarm_quat_min_norm = float(xarm_quat_norm)
        else:
            self._xarm_quat_min_norm = min(self._xarm_quat_min_norm, float(xarm_quat_norm))
        if xarm_quat_norm < XARM_MIN_QUAT_NORM:
            if self._last_valid_xarm_pose is None:
                raise RuntimeError(
                    f"xarm quaternion norm too small ({xarm_quat_norm:.4f}) "
                    "and no fallback pose is available."
                )
            xarm_pose = self._last_valid_xarm_pose.copy()
            self._last_xarm_action_fallback = {
                "reason": "xarm_quat_norm_too_small",
                "xarm_quat_norm": float(xarm_quat_norm),
            }
            self._xarm_quat_fallback_count += 1
            if self._xarm_quat_fallback_first_step is None:
                self._xarm_quat_fallback_first_step = getattr(self, "env_step", None)
            print(
                f"Bad xarm quaternion norm {xarm_quat_norm:.4f} "
                f"at env_step={getattr(self, 'env_step', None)}; reuse previous xarm pose."
            )
        else:
            xarm_quat /= xarm_quat_norm
            xarm_rpy = R.from_quat(xarm_quat).as_euler("xyz", degrees=False).astype(np.float32)
            xarm_pose = np.concatenate([action[0:3], xarm_rpy], axis=0).astype(np.float32)
            self._last_valid_xarm_pose = xarm_pose.copy()
            self._last_xarm_action_fallback = None
        franka_quat /= max(np.linalg.norm(franka_quat), 1e-12)

        franka_rotvec = R.from_quat(franka_quat).as_rotvec().astype(np.float32)

        return np.concatenate(
            [
                xarm_pose,
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
