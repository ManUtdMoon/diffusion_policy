import time
import numpy as np
import cv2
from pynput import keyboard
from gym import spaces
from scipy.spatial.transform import Rotation as R

from diffusion_policy.env.box.xarm_wrapper import XArmWrapper
from diffusion_policy.env.box.franka_wrapper import FrankaWrapper
from diffusion_policy.env.box.robotiq_wrapper import RobotiqWrapper
from diffusion_policy.env.box.realsense_multiview import MultiViewRealSense, DEFAULT_MULTI_CAMERAS
from diffusion_policy.env.flip.franka.common.precise_sleep import precise_wait

is_done = False
def on_press(key):
    global is_done
    try:
        if hasattr(key, 'char') and key.char:
            ch = key.char
            if ch.isdigit() or ch.isalpha() or ch in ['+', '-']:
                is_done = True
        elif hasattr(key, 'vk') and key.vk == 65437:
            is_done = True
    except AttributeError:
        # Ignore special keys that don't carry a `char` attribute.
        pass

keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()


class BoxEnv:
    """
    Env for deploying policy on the box task.

    Action layout (14, same as teleop.py):
    [xarm_pos(3,m), xarm_rpy(3,rad), xarm_gripper(1),
     franka_pos(3,m), franka_rotvec(3,rad), franka_gripper(1)]

    Observation:
    - qpos: (15,) = [xarm_q(6), xarm_gripper(1), franka_q(7), franka_gripper(1)]
    - images: dict(cam_name -> (3,128,128), float32 in [0,1], RGB)
    """

    def __init__(
        self,
        dt=1 / 20.,
        camera_cfg=None,
    ):
        self.dt = dt
        self.camera_cfg = camera_cfg if camera_cfg is not None else DEFAULT_MULTI_CAMERAS

        # Match teleop defaults.
        self.xarm = XArmWrapper(joints_init=[-47.6, -9.7, -73.6, -3.1, 82.5, -43.0])
        self.xarm_gripper = RobotiqWrapper(robot="xarm")
        self.franka = FrankaWrapper(joints_init=(1.0173, -0.4755, 0.0787, -2.4569, 0.0160, 1.9935, 1.1648))
        self.franka_gripper = RobotiqWrapper(robot="franka")

        self.camera = MultiViewRealSense(self.camera_cfg)
        self.camera.start()

        # Build observation/action spaces.
        # Primary deployment format is 16-dim quat action:
        # [xarm_pos3, xarm_quat4, xarm_gripper1, franka_pos3, franka_quat4, franka_gripper1]
        # step() also accepts legacy 14-dim action directly.
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
                "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32),
                "global": spaces.Box(low=0.0, high=1.0, shape=(3, 128, 128), dtype=np.float32),
                "wrist_0": spaces.Box(low=0.0, high=1.0, shape=(3, 128, 128), dtype=np.float32),
                "wrist_1": spaces.Box(low=0.0, high=1.0, shape=(3, 128, 128), dtype=np.float32),
        })

        print("BoxEnv initialized.")

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
        xarm_cmd[:3] *= 1000.0  # m -> mm
        xarm_cmd[3:] = xarm_cmd[3:] * 180.0 / np.pi  # rad -> deg
        self.xarm.set_servo_cartesian(xarm_cmd)
        if action14[6] > 0.5:
            self.xarm_gripper.close()
        else:
            self.xarm_gripper.open()

        # Franka
        franka_target = np.array(action14[7:13], dtype=np.float32)
        self.franka.franka.servoL(franka_target, self.dt)
        if action14[13] > 0.5:
            self.franka_gripper.close()
        else:
            self.franka_gripper.open()

    def reset(self):
        print("<RESET>")
        global is_done
        self.env_step = 0
        self.done = False
        is_done = False

        # reset robot
        self.xarm.reset()
        self.xarm_gripper.close()
        self.franka.franka.reset_home()
        self.franka_gripper.close()
        time.sleep(2.0)
        input("Press Enter to continue...")

        obs = self._get_obs()

        self.t_start = time.monotonic()
        return obs

    def reset_end(self):
        pass

    def step(self, action):
        global is_done
        self.env_step += 1
        t_cycle_end = self.t_start + self.env_step * self.dt

        if not self.done:
            # control the robot
            action14 = self._transform_action(action)
            self._apply_action(action14)

        precise_wait(t_cycle_end)

        obs = self._get_obs()

        self.done, timeout = self.terminate(is_done)

        # post-process
        reward = 0.0
        is_success = False
        if self.done:
            user_input = input(
                "\nEpisode ended. [0-9]=success, [a-z]=failure, [+]=s+regrasp, [-]=f+regrasp: "
            ).strip()
            if user_input:
                label = user_input[-1]

            is_success = label.isdigit()
            reward = 1.0 if is_success else 0.0
            print(f"{'Success' if is_success else 'Failure'} recorded!")

        return obs, reward, self.done, {'is_success': is_success, 'timeout': timeout,}

    def terminate(self, is_done):
        '''
        return done, timeout
        '''
        if is_done:
            return True, False
        else:
            if self.env_step >= 400:
                return True, True
            else:
                return False, False

    def close(self):
        try:
            self.xarm_gripper.open()
            self.franka_gripper.open()
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
            rgb = rgb[:, 66:306]
        resized_img = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
        img = resized_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return img

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

        action_14 = np.concatenate(
            [
                action[0:3],         # xarm pos
                xarm_rpy,            # xarm rpy
                action[7:8],         # xarm gripper
                action[8:11],        # franka pos
                franka_rotvec,       # franka rotvec
                action[15:16],       # franka gripper
            ],
            axis=0,
        ).astype(np.float32)
        return action_14
