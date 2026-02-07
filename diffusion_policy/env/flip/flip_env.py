import time
import numpy as np
import cv2
import threading
import queue
from copy import deepcopy
from pynput import keyboard
from gym import spaces
from scipy.spatial.transform import Rotation as R

from diffusion_policy.env.flip.franka.franka_wrapper import FrankaWrapper
from diffusion_policy.env.flip.realsense_flip import RealSense
from diffusion_policy.env.flip.franka.common.precise_sleep import precise_wait
from diffusion_policy.env.flip.joints import START_JOINTS
    

is_done = False
need_regrasp = False
def on_press(key):
    global is_done
    try:
        if (
            (hasattr(key, 'vk') and key.vk == 65437) or
            (key.char >= '0' and key.char <= '9')
        ):
            is_done = True
        elif hasattr(key, 'vk') and key.char >= 'a' and key.char <= 'z':
            is_done = True
    except AttributeError:
        # Ignore special keys that don't carry a `char` attribute.
        pass

keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()    


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

        self.dt = dt
        self.camera = RealSense(color_width=320, color_height=240)
        self.camera.start()
        self.camera_thread = threading.Thread(target=self._get_camera_frame)
        self.camera_thread.start()
        self.camera_queue = queue.Queue(maxsize=1)

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

        # reser robot
        self.franka.franka.reset_home()
        time.sleep(3.0)
        input("Press Enter to continue...")

        # get current observation
        ## image
        frame = self.get_frame()
        raw_img = frame['color']  # (H, W, 3(bgr)), uint8

        ## low_dim
        qpos = self.franka.get_joint()
        obs = {
            'qpos': np.array(qpos, dtype=np.float32),
            'image': self._transform_image(raw_img),
        }

        self.t_start = time.monotonic()
        self.prev_target = self.franka.get_pose()
        return obs.copy()

    def regrasp(self):
        pass

    def reset_end(self):
        self.regrasp()

    def step(self, action):
        print(action)
        global is_done
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

        ## low_dim
        qpos = self.franka.get_joint()
        obs = {
            'qpos': np.array(qpos, dtype=np.float32),
            'image': self._transform_image(raw_img),
        }

        self.done, timeout = self.terminate(is_done)

        # post-process
        reward = 0.0
        is_success = False
        if self.done:
            user_input = input("\nEpisode ended. Type number for success, letter for failure:").strip()
            # Use the last character to avoid interference from stop key
            if user_input:
                user_input = user_input[-1]
            if user_input and user_input[0].isdigit():
                is_success = True
                reward = 1.
                print(f"Success recorded!")
            else:
                # Record as failure  
                print(f"Failure recorded!")
                is_success = False
                reward = 0.

        return obs, reward, self.done, {'is_success': is_success, 'timeout': timeout,}

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