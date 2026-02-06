import numpy as np
from multiprocessing.managers import SharedMemoryManager
from scipy.spatial.transform import Rotation as R
from .franka_interpolation_controller import FrankaInterpolationController


class FrankaWrapper:
    def __init__(self, joints_init):
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.franka = FrankaInterpolationController(
            shm_manager=self.shm_manager,
            robot_ip='172.16.0.1',
            frequency=100,
            Kx_scale=5.0,
            Kxd_scale=2.0,
            joints_init=joints_init,
            verbose=False,
        )
        self.franka.start()
        
    def get_pose(self):
        state = self.franka.get_state()
        pose = np.zeros(6, dtype=np.float32)
        pose[:3] = state['ActualTCPPose'][:3]
        pose[3:] = state['ActualTCPPose'][3:6]
        return pose

    def get_joint(self):
        state = self.franka.get_state()
        return state['ActualQ']

    def close(self):
        self.franka.kill()
        self.franka.join()
        self.shm_manager.shutdown()
