import time
import numpy as np
import threading
from xarm.wrapper import XArmAPI
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def pose_interp(pose1, pose2, alpha):
    # pose: [x, y, z, roll, pitch, yaw]，角度单位为deg
    pos1 = np.array(pose1[:3])
    pos2 = np.array(pose2[:3])
    pos_interp = pos1 + (pos2 - pos1) * alpha

    # 欧拉角转四元数
    rot1 = R.from_euler('xyz', pose1[3:], degrees=True)
    rot2 = R.from_euler('xyz', pose2[3:], degrees=True)
    # Slerp插值
    slerp = Slerp([0, 1], R.concatenate([rot1, rot2]))
    quat_interp = slerp([alpha])[0]
    euler_interp = quat_interp.as_euler('xyz', degrees=True)
    return np.concatenate([pos_interp, euler_interp])


class XArmWrapper:
    def __init__(self, joints_init, ip='192.168.1.202'):
        self.xarm = XArmAPI(ip)
        self.joints_init = joints_init
        self.first_reset = True
        self._running = False
        self._lock = threading.Lock()
        self.reset()
    
    def reset(self):
        if not self.first_reset:
            self.close()
        self.xarm.set_mode(0)
        self.xarm.set_state(0)
        self.xarm.set_servo_angle(angle=self.joints_init, speed=32, wait=True)
        curr_pos = np.array(self.xarm.get_position()[1])
        self._target = curr_pos.copy()
        self._last_target = curr_pos.copy()
        self._interp_steps = 10
        self._interp_count = 0
        self._running = True
        self._thread = threading.Thread(target=self._control_loop)
        self._thread.start()
        self.first_reset = False

    def set_servo_cartesian(self, target):
        with self._lock:
            self._last_target = np.array(self.xarm.get_position()[1])
            self._target = np.array(target)
            self._interp_count = 0
    
    def _control_loop(self):
        self.xarm.set_mode(1)
        self.xarm.set_state(state=0)
        control_freq = 200  # Hz
        dt = 1.0 / control_freq
        while self._running:
            with self._lock:
                if self._interp_count < self._interp_steps:
                    alpha = (self._interp_count + 1) / self._interp_steps
                    interp = pose_interp(self._last_target, self._target, alpha)
                    self._interp_count += 1
                else:
                    interp = self._target
            self.xarm.set_servo_cartesian(interp)
            time.sleep(dt)

    def get_position(self):
        code, pose = self.xarm.get_position()
        if code != 0:
            raise RuntimeError("Abnormal code returned by XArm!")
        pose = np.array(pose)
        pose[:3] /= 1000  # mm -> m
        pose[3:] = pose[3:] / 180 * np.pi
        return pose
    
    def get_joint(self):
        code, angle = self.xarm.get_servo_angle()
        return np.array(angle) / 180 * np.pi

    def close(self):
        self._running = False
        self._thread.join()
