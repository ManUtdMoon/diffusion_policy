import numpy as np
import time
import threading
from multiprocessing.managers import SharedMemoryManager
from scipy.spatial.transform import Rotation as R

from diffusion_policy.env.box.franka.franka_interpolation_controller import FrankaInterpolationController


class FrankaWrapper:
    def __init__(self, joints_init):
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.joints_init = joints_init
        self.lock = threading.Lock()
        
        self._create_franka_controller()
        self._start_franka()
    
    def _create_franka_controller(self):
        """创建Franka控制器实例"""
        print("[FrankaWrapper] Creating FrankaInterpolationController...")
        try:
            self.franka = FrankaInterpolationController(
                shm_manager=self.shm_manager,
                robot_ip='172.16.0.1',
                frequency=100,
                Kx_scale=2.5,
                Kxd_scale=1.0,
                joints_init=self.joints_init,
                verbose=False,  # 启用详细日志
            )
            print("[FrankaWrapper] FrankaInterpolationController created successfully")
        except Exception as e:
            print(f"[FrankaWrapper] Failed to create FrankaInterpolationController: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _start_franka(self):
        """启动Franka控制器"""
        with self.lock:
            try:
                print("[FrankaWrapper] Starting Franka process...")
                self.franka.start()
                print("[FrankaWrapper] Franka process started, waiting for ready...")
                
                # 等待进程准备就绪
                max_wait_time = 10.0
                start_time = time.time()
                while not self.franka.is_ready and time.time() - start_time < max_wait_time:
                    time.sleep(0.1)
                
                if self.franka.is_ready:
                    print("[FrankaWrapper] Franka controller started successfully")
                else:
                    print("[FrankaWrapper] WARNING: Franka controller not ready after timeout")
                    
            except Exception as e:
                print(f"[FrankaWrapper] Failed to start Franka controller: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to start Franka controller: {e}")
    
    def restart(self):
        """重启Franka控制器"""
        with self.lock:
            try:
                print("[FrankaWrapper] Stopping Franka controller...")
                if hasattr(self, 'franka'):
                    if self.franka.is_alive():
                        print("[FrankaWrapper] Stopping Franka process...")
                        self.franka.stop()
                        time.sleep(3.0)  # 等待进程完全停止
                        
                        # 强制终止进程（如果还在运行）
                        if self.franka.is_alive():
                            print("[FrankaWrapper] Force terminating Franka process...")
                            self.franka.terminate()
                            time.sleep(1.0)
                            if self.franka.is_alive():
                                print("[FrankaWrapper] Force killing Franka process...")
                                self.franka.kill()
                                time.sleep(1.0)
                    
                    # 确保进程已经完全停止
                    if self.franka.is_alive():
                        print("[FrankaWrapper] WARNING: Franka process still alive after cleanup!")
                        return
                    
                    # 清理旧的控制器引用
                    del self.franka
                
                # 强制垃圾回收，确保进程完全清理
                import gc
                gc.collect()
                time.sleep(1.0)  # 增加等待时间
                
                # 重新创建共享内存管理器
                print("[FrankaWrapper] Recreating shared memory manager...")
                try:
                    self.shm_manager.shutdown()
                    time.sleep(1.0)
                except:
                    pass
                
                self.shm_manager = SharedMemoryManager()
                self.shm_manager.start()
                time.sleep(1.0)
                
                print("[FrankaWrapper] Creating new Franka controller...")
                # 重新创建控制器
                self._create_franka_controller()
                
                print("[FrankaWrapper] Starting new Franka controller...")
                self._start_franka()
                
                print("[FrankaWrapper] Franka controller restarted successfully")
                
            except Exception as e:
                print(f"[FrankaWrapper] Failed to restart Franka controller: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to restart Franka controller: {e}")
    
    def is_alive(self):
        """检查Franka进程是否存活"""
        return self.franka.is_alive()
    
    def get_tcp_pose(self):
        """获取TCP位姿"""
        if not self.franka.is_alive():
            raise RuntimeError("Franka process is not alive")
        state = self.franka.get_state()
        tcp_pose = state['ActualTCPPose']
        
        # 数值稳定性检查
        if np.any(np.isnan(tcp_pose)) or np.any(np.isinf(tcp_pose)):
            print("[WARNING] Franka TCP pose contains NaN or Inf values")
            # 返回零向量作为安全值
            return np.zeros(6, dtype=np.float32)
        
        # 检查位置和旋转的合理性
        pos_norm = np.linalg.norm(tcp_pose[:3])
        if pos_norm > 10.0:  # 如果位置超过10米，可能是异常值
            print(f"[WARNING] Franka TCP position too large: {pos_norm:.3f}m")
            return np.zeros(6, dtype=np.float32)
        
        return tcp_pose
        
    def get_joint(self):
        """获取关节角度"""
        if not self.franka.is_alive():
            raise RuntimeError("Franka process is not alive")
        state = self.franka.get_state()
        joint_angles = state['ActualQ']
        
        # 数值稳定性检查
        if np.any(np.isnan(joint_angles)) or np.any(np.isinf(joint_angles)):
            print("[WARNING] Franka joint angles contain NaN or Inf values")
            # 返回初始关节角度作为安全值
            return np.array(self.joints_init, dtype=np.float32)
        
        # 检查关节角度的合理性（Franka关节限制）
        joint_limits = [
            (-2.8973, 2.8973),   # joint 1
            (-1.7628, 1.7628),   # joint 2  
            (-2.8973, 2.8973),   # joint 3
            (-3.0718, -0.0698),  # joint 4
            (-2.8973, 2.8973),   # joint 5
            (-0.0175, 3.7525),   # joint 6
            (-2.8973, 2.8973),   # joint 7
        ]
        
        for i, (angle, (min_angle, max_angle)) in enumerate(zip(joint_angles, joint_limits)):
            if angle < min_angle or angle > max_angle:
                print(f"[WARNING] Franka joint {i+1} angle {angle:.3f} out of range [{min_angle:.3f}, {max_angle:.3f}]")
                return np.array(self.joints_init, dtype=np.float32)
        
        return joint_angles
    
    def reset(self):
        """重置到初始位置"""
        if not self.franka.is_alive():
            raise RuntimeError("Franka process is not alive")
        self.franka.reset_home()
    
    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            if hasattr(self, 'franka') and self.franka.is_alive():
                self.franka.stop()
            if hasattr(self, 'shm_manager'):
                self.shm_manager.shutdown()
        except:
            pass

    def close(self):
        try:
            if hasattr(self, 'franka') and self.franka.is_alive():
                self.franka.stop()
            if hasattr(self, 'shm_manager'):
                self.shm_manager.shutdown()
        except:
            pass
