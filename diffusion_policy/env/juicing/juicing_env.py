import time
from math import pi
import numpy as np
import cv2
from gym import spaces
import threading
import queue
from pynput import keyboard

from diffusion_policy.env.juicing.xarm_wrapper import XArmWrapper
from diffusion_policy.env.juicing.robotiq_wrapper import RobotiqWrapper
from diffusion_policy.env.juicing.realsense import RealSense

SMALL_SIZE = (320, 180)
RESIZE_SIZE = 128


def apply_mask(image: np.ndarray) -> np.ndarray:
    """Apply mask to image: zero out all pixels above the line connecting (127,0) and (0,45).

    The line equation: W = (-45/127) * H + 45
    Pixels where W < line_W are masked (set to 0).

    Args:
        image: Input image with shape (H, W, C) or (N, H, W, C), in RGB format.

    Returns:
        Masked image with same shape as input.
    """
    H, W = image.shape[-3:-1]

    # Create coordinate grids
    h_coords = np.arange(H)
    w_coords = np.arange(W)
    hh, ww = np.meshgrid(h_coords, w_coords, indexing='ij')  # hh[H, W], ww[H, W]

    # Line: passes through (H=127, W=0) and (H=0, W=45)
    # Slope: (45 - 0) / (0 - 127) = -45/127
    # W - 0 = (-45/127) * (H - 127)
    # W = (-45/127) * H + 45
    # Mask: W < (-45/127) * H + 45  -> above/left of the line
    line_w = (-45 / 127) * hh + 45
    mask = ww < line_w  # [H, W] boolean, True = mask out

    if image.ndim == 3:
        # H, W, C
        result = image.copy()
        result[mask, :] = 0
        return result
    elif image.ndim == 4:
        # N, H, W, C
        result = image.copy()
        result[:, mask, :] = 0
        return result
    else:
        raise ValueError(f"Unexpected image ndim: {image.ndim}")


def transform_image(image: np.ndarray) -> np.ndarray:
    """Transform raw camera image to processed observation image.

    Processing pipeline:
    1. Crop: [50:, 7:] on H, W dimensions
    2. Convert BGR to RGB
    3. Resize to resize_size x resize_size
    4. Apply mask to zero out pixels above diagonal line

    Args:
        image: Input image with shape (H, W, C), BGR format from camera.
        resize_size: Output image size (default: 128).

    Returns:
        Processed image with shape (C, H, W), RGB format, float32, range [0, 1].
    """
    # Crop: H from 50, W from 7
    small = cv2.resize(image, SMALL_SIZE).astype(np.uint8)
    cropped = small[70:, 10:290, ::-1] # bgr -> rgb

    # # Resize
    # resized = cv2.resize(rgb, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)

    # # Apply mask
    # masked = apply_mask(resized)

    # cv2.imshow("Masked Image", masked[..., ::-1])  # Show in BGR format for OpenCV
    # cv2.waitKey(1)

    # Convert to CHW format and normalize to [0, 1]
    result = np.moveaxis(cropped.astype(np.float32) / 255., -1, 0)

    return result


is_done = False
def on_press(key):
    global is_done, is_success
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


def precise_sleep(dt: float, slack_time: float=0.001, time_func=time.monotonic):
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return


def precise_wait(t_end: float, slack_time: float=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return


class JuicingEnv:
    INIT_JOINTS = np.array([-91.4, 10.6, -18.8, -276.6, 76.4, -79.1])  # Initial joint positions in degrees 2, parallel
    END_JOINTS = np.array([-47.4, -2.8, -32.3, -287.6, 107.2, 128])   # End joint positions in degrees 2
    # INIT_JOINTS = np.array([-16.3, 2, -85.7, -323.8, 95.1, 74.1])  # Initial joint positions in degrees for stage 2
    # END_JOINTS = np.array([-22.9, -41.2, -49, -316.5, 108.1, -18.8])   # End joint positions in degrees 2 for stage 2

    def __init__(
        self, 
        robot_ip='192.168.1.202',
        dt=1/30,
    ):
        print("Juicing Env Init")
        
        # Initialize XArm and gripper
        self.xarm = XArmWrapper(joints_init=self.INIT_JOINTS, ip=robot_ip)
        self.gripper = RobotiqWrapper(robot='xarm')

        self.dt = dt

        self._cur_qpos = None
        self._cur_tcp = None  # (xyz, euler)

        # Initialize camera
        self.camera = RealSense()
        self.camera.start()
        self.camera_thread = threading.Thread(target=self._get_camera_frame)
        self.camera_thread.start()
        self.camera_queue = queue.Queue(maxsize=1)
        self.pre_action = None
        
        # Environment parameters
        obs_sensor_dim = 7  # 6 DOF pose + 1 gripper state
        act_dim = 8         # 6 DOF pose + 1 gripper action + 1 is_done flag (optional)

        # Action space: [x, y, z, r, p, yaw, gripper, is_done]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(3, 128, 128),
                dtype=np.uint8
            ),
            'qpos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_sensor_dim,),
                dtype=np.float32
            ),
        })

        print("Juicing Env Init Done")

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
        global is_done
        print("<RESET>")
        self.env_step = 0
        self.done = False
        is_done = False

        # Reset robot to initial position
        self.xarm.reset()
        self.gripper.open()
        time.sleep(0.5)

        # Press enter to continue
        input("Press Enter to continue after resetting the robot...")

        # Get current robot state
        xarm_pose = self.xarm.get_position()  # [x, y, z, roll, pitch, yaw] in meters and radians
        xarm_q = self.xarm.get_joint()      # Joint angles in degrees
        gripper_state = self.gripper.get_state()

        # Get camera frame and process image
        frame = self.get_frame()
        color_img = transform_image(frame['color'])

        # Agent position: [x, y, z, roll, pitch, yaw, gripper_state]
        q_pos = np.concatenate([xarm_q, [gripper_state]])
        ee_pose = np.concatenate([xarm_pose, [gripper_state]])

        obs = {
            'qpos': np.array(q_pos),
            'image': color_img,
            'ee_pose': np.array(ee_pose),
        }

        self.t_start = time.monotonic()
        return obs.copy()


    def reset_end(self):
        print("<RESET END>")
        # Stop the control loop temporarily
        self.xarm._running = False
        if hasattr(self.xarm, '_thread'):
            self.xarm._thread.join()

        # Switch to position control mode
        self.xarm.xarm.set_mode(0)
        self.xarm.xarm.set_state(0)

        # Pad END_JOINTS to 7-DOF (already in degrees)
        end_joints_7dof = np.append(self.END_JOINTS, 0)

        # Move to end position
        current_joints_7dof = self.xarm.xarm.get_servo_angle()[1]
        out_joints_7dof = end_joints_7dof.copy()
        out_joints_7dof[5] = current_joints_7dof[5]
        self.xarm.xarm.set_servo_angle(angle=out_joints_7dof, speed=128, wait=True)
        self.xarm.xarm.set_servo_angle(angle=end_joints_7dof, speed=128, wait=True)

        # Update position but don't restart control loop
        # The next episode's reset() will handle that
        curr_pos = np.array(self.xarm.xarm.get_position()[1])
        self.xarm._target = curr_pos.copy()
        self.xarm._last_target = curr_pos.copy()
        self.xarm.first_reset = False  # Mark that we've done at least one reset

        # Press enter to continue
        self.gripper.open()
        input("Press Enter to continue after resetting the robot...")


    def step(self, action):
        global is_done
        start_time = time.monotonic()
        self.env_step += 1
        t_cycle_end = self.t_start + self.env_step * self.dt
        
        # Get current robot state (like in teleop.py lines 165-172)
        xarm_q = self.xarm.get_joint()
        xarm_pose = self.xarm.get_position()
        xarm_gripper_state = self.gripper.get_state()

        q_pos = np.concatenate([xarm_q, [xarm_gripper_state]])
        ee_pose = np.concatenate([xarm_pose, [xarm_gripper_state]])

        if not self.done:
            # Action format: [x, y, z, roll, pitch, yaw, gripper] - absolute target position
            # XArm target setup (following teleop.py lines 177-181)
            xarm_target = np.zeros(6, dtype=np.float32)
            xarm_target[:3] = action[:3] * 1000  # Position in mm (like teleop.py line 179)
            xarm_target[3:] = action[3:6] * 180 / pi  # Orientation in degrees (like teleop.py line 180)

            # Send XArm cartesian command (like teleop.py line 181)
            self.xarm.set_servo_cartesian(xarm_target)

            # Gripper control (following teleop.py lines 185-196)
            # action[6] is normalized gripper value (0-1)
            # Convert to gripper position (0-255), reverse mapping like teleop.py line 193
            gripper_target = int((1.0 - action[6]) * 255)
            self.gripper.set_position(gripper_target)

        # Precise timing control like teleop.py line 225
        precise_wait(t_cycle_end)

        # Get updated robot state after action
        xarm_q = self.xarm.get_joint()
        xarm_pose = self.xarm.get_position()
        xarm_gripper_state = self.gripper.get_state()

        q_pos = np.concatenate([xarm_q, [xarm_gripper_state]])
        ee_pose = np.concatenate([xarm_pose, [xarm_gripper_state]])
        
        frame = self.get_frame()
        color_img = transform_image(frame['color'])

        obs = {
            'qpos': np.array(q_pos),
            'image': color_img,
            'ee_pose': np.array(ee_pose),
        }
        if action.shape[-1] == 8:
            self.done, timeout = self.terminate(is_done, action[7] > 0.5)
            # print("Predict done value:", action[7], "Predicted done:", action[7] > 0.5)
        else:
            self.done, timeout = self.terminate(is_done)

        reward = 0.
        return_success = False
        if self.done:
            self.gripper.set_position(0)
            time.sleep(0.5)

            print("\nEpisode ended. Was it successful?")
            user_input = input("Type num for success, letter for failure: ").strip()
            # Use the last character to avoid interference from stop key
            if user_input:
                user_input = user_input[-1]
            if user_input and user_input[0].isdigit():
                return_success = True
                reward = 1.
                print(f"Success recorded!")
            else:
                print(f"Failure recorded!")

            # Reset manual flags for the next episode.
            is_done = False

        return obs.copy(), reward, self.done, {'is_success': return_success, 'timeout': timeout}


    def terminate(self, is_done, predict_done=None):
        if is_done or predict_done == True:
            return True, False
        else:
            if self.env_step >= 450:
                return True, True
            else:
                return False, False

    def close(self):
        print("Closing Juicing Environment")
        self.xarm.close()
        self.gripper.close()
        self.camera.stop()

    def render(self, mode='rgb_array'):
        if hasattr(self, 'camera_queue') and not self.camera_queue.empty():
            frame = self.camera_queue.queue[0]
            return frame['color']
        else:
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def __del__(self):
        try:
            self.close()
        except:
            pass