import time
import numpy as np
import cv2
import threading
import queue
from pynput import keyboard
from gym import spaces
from scipy.spatial.transform import Rotation as R
from pyrobotiqgripper import RobotiqGripper

from diffusion_policy.env.flip.franka.franka_wrapper import FrankaWrapper
from diffusion_policy.env.flip.realsense_flip import RealSense
from diffusion_policy.env.flip.franka.common.precise_sleep import precise_wait
from diffusion_policy.env.flip.joints import (
    START_JOINTS,
    PRE_GRASP_JOINTS,
    GRASP_JOINTS,
    LIFT_JOINTS,
)
    

is_done = False
need_regrasp = False
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


def bgr_hwc_to_rgb_chw_uint8(image: np.ndarray) -> np.ndarray:
    """Convert camera frame from HWC BGR uint8 to CHW RGB uint8."""
    return np.moveaxis(image[:, :, ::-1], -1, 0).copy()


class FlipEnv:
    def __init__(
        self, 
        dt=1 / 10,
        mode='rel', # 'rel' or 'abs'
        lookahead_steps=3,
        smoothing_weight=0.7
    ):
        self.franka = FrankaWrapper(
            joints_init=START_JOINTS,
            lookahead_steps=lookahead_steps,
            smoothing_weight=smoothing_weight,
        )
        self.gripper = None

        self.dt = dt
        self.camera = RealSense(color_width=320, color_height=240)
        self.camera.start()
        self.camera_queue = queue.Queue(maxsize=1)
        self.camera_thread = threading.Thread(target=self._get_camera_frame)
        self.camera_thread.start()
        self._raw_step_images = []
        self._raw_step_timestamps = []

        self.mode = mode
        self.prev_target = None
        assert mode in ['rel', 'abs'], f"mode should be 'rel' or 'abs', got {mode}."

        obs_sensor_dim = 7
        act_dim = 6

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3, 128, 128),
                dtype=np.float32
            ),
            'qpos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_sensor_dim,),
                dtype=np.float32
            ),
        })

        print("Franka Env Init Done")

    def start_dense_recording(self):
        self._raw_step_images = []
        self._raw_step_timestamps = []

    def stop_dense_recording(self):
        return

    def get_dense_recording(self):
        if len(self._raw_step_images) == 0:
            images = np.array([])
        else:
            images = np.stack(self._raw_step_images, axis=0)
        timestamps = np.array(self._raw_step_timestamps, dtype=np.float64)
        return {
            'image': images,
            'timestamp': timestamps
        }

    def _ensure_gripper(self):
        if self.gripper is None:
            self.gripper = RobotiqGripper(portname='/dev/ttyUSB0')
        return self.gripper

    def _get_camera_frame(self):
        while True:
            frame = self.camera.get_frame()
            if self.camera_queue.full():
                self.camera_queue.get()
            self.camera_queue.put(frame)

    def get_frame(self):
        while True:
            if self.camera_queue.full():
                frame = self.camera_queue.get()
                assert not self.camera_queue.full()
                return frame
            time.sleep(1 / 300)

    def reset(self):
        print("<RESET>")
        global is_done
        self.env_step = 0
        self.done = False
        is_done = False

        # reset robot
        self.franka.franka.reset_home()
        time.sleep(2.0)
        input("Press Enter to continue...")

        # get current observation
        ## image
        frame = self.get_frame()
        raw_img = frame['color']  # (H, W, 3(bgr)), uint8
        raw_img_chw = bgr_hwc_to_rgb_chw_uint8(raw_img)

        ## low_dim
        qpos = self.franka.get_joint()
        obs = {
            'qpos': np.array(qpos, dtype=np.float32),
            'image': self._transform_image(raw_img),
        }

        self.start_dense_recording()
        self._raw_step_images.append(raw_img_chw)
        self._raw_step_timestamps.append(time.monotonic())
        self.t_start = time.monotonic()
        self.prev_target = self.franka.get_pose()
        return obs.copy()

    def regrasp(self):
        print("<REGRASP>")
        gripper = self._ensure_gripper()
        gripper.goTo(100)
        time.sleep(0.5)

        print("Moving to pre_grasp...")
        self.franka.franka.servoJ(np.array(PRE_GRASP_JOINTS), from_cartesian=True)

        print("Moving to grasp...")
        self.franka.franka.servoJ(np.array(GRASP_JOINTS), from_cartesian=False)
        time.sleep(1.0)
        input("Place the spatula and press Enter to continue...")    

        print("Closing gripper...")
        gripper.close(speed=128)
        time.sleep(0.5)

        print("Lifting...")
        self.franka.franka.servoJ(np.array(LIFT_JOINTS), from_cartesian=False)

    def reset_end(self):
        global need_regrasp
        if need_regrasp:
            need_regrasp = False
            self.regrasp()

    def step(self, action):
        global is_done, need_regrasp
        self.env_step += 1
        t_cycle_end = self.t_start + self.env_step * self.dt

        if not self.done:
            # control the robot
            target = self._action_to_pose(action, self.prev_target)
            self.franka.franka.servoL(target, self.dt)
            self.prev_target = target

        precise_wait(t_cycle_end)

        # get current observation
        ## image
        frame = self.get_frame()
        raw_img = frame['color']  # (H, W, 3(bgr)), uint8
        raw_img_chw = bgr_hwc_to_rgb_chw_uint8(raw_img)

        ## low_dim
        qpos = self.franka.get_joint()
        obs = {
            'qpos': np.array(qpos, dtype=np.float32),
            'image': self._transform_image(raw_img),
        }
        self._raw_step_images.append(raw_img_chw)
        self._raw_step_timestamps.append(time.monotonic())

        self.done, timeout = self.terminate(is_done)

        # post-process
        reward = 0.0
        is_success = False
        if self.done:
            self.stop_dense_recording()
            user_input = input(
                "\nEpisode ended. [0-9]=success, [a-z]=failure, [+]=s+regrasp, [-]=f+regrasp: "
            ).strip()
            if user_input:
                label = user_input[-1]

            is_success, need_regrasp = self._parse_episode_label(label)
            reward = 1.0 if is_success else 0.0

            if is_success:
                print(f"Success recorded! need_regrasp={need_regrasp}")
            else:
                print(f"Failure recorded! need_regrasp={need_regrasp}")

        return obs, reward, self.done, {'is_success': is_success, 'timeout': timeout,}

    def _parse_episode_label(self, label):
        if label is None:
            return False, False

        if label == '+':
            return True, True
        if label == '-':
            return False, True
        if label.isdigit():
            return True, False
        if label.isalpha():
            return False, False
        return False, False

    def terminate(self, is_done):
        '''
        return done, timeout
        '''
        if is_done:
            return True, False
        else:
            if self.env_step >= 250:
                return True, True
            else:
                return False, False

    def close(self):
        self.camera.stop()
        self.franka.close()

    def _transform_image(self, raw_img):
        # crop, reverse channel, resize, transpose, dtype
        crop_img = raw_img[0:165, 40:280, ::-1]
        resized_img = cv2.resize(crop_img, (128, 128), interpolation=cv2.INTER_LINEAR)
        img = resized_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return img

    def _action_to_pose(self, action, prev_target):
        if self.mode == 'rel':
            target = np.zeros_like(prev_target)
            target[:3] = prev_target[:3] + action[:3] # pos
            target[3:] = (
                R.from_rotvec(action[3:]) * R.from_rotvec(prev_target[3:])
            ).as_rotvec()  # rot
        elif self.mode == 'abs':  # abs
            target = action
        else:
            raise NotImplementedError(f"mode should be 'rel' or 'abs', got {self.mode}.")
        return target
