import time
import threading
from pyrobotiqgripper import RobotiqGripper

class RobotiqWrapper:
    def __init__(self, robot):
        portname='/dev/ttyUSB0' if robot == 'franka' else '/dev/ttyUSB1'
        
        # 检查设备是否存在
        import os
        if not os.path.exists(portname):
            print(f"[WARNING] Robotiq gripper port {portname} does not exist!")
            self.gripper = None
            self.current_state = 'close'
            self.change_state = False
            return
        
        try:
            self.gripper = RobotiqGripper(portname=portname)
            # self.gripper.reset()
            print(f"[RobotiqWrapper] Activating gripper on {portname}...")
            self.gripper.activate()
            print(f"[RobotiqWrapper] Gripper activated successfully")
            
            self.current_state = 'close'
            self.change_state = True
            
            self.gripper_thread = threading.Thread(target=self._monitor_gripper)
            self.gripper_thread.daemon = True
            self.gripper_thread.start()
            
        except Exception as e:
            print(f"[WARNING] Failed to initialize Robotiq gripper: {e}")
            self.gripper = None
            self.current_state = 'open'
            self.change_state = False

    def _monitor_gripper(self):
        while True:
            if self.gripper is not None and self.change_state:
                try:
                    if self.current_state == 'open':
                        self.gripper.goTo(96)
                    elif self.current_state == 'close':
                        self.gripper.close()
                except Exception as e:
                    print(f"[WARNING] Gripper control error: {e}")
            time.sleep(1 / 30)

    def open(self):
        if self.current_state != 'open':
            self.current_state = 'open'
            self.change_state = True
            if self.gripper is None:
                print("[WARNING] Cannot open gripper - gripper not initialized")
    
    def close(self):
        if self.current_state != 'close':
            self.current_state = 'close'
            self.change_state = True
            if self.gripper is None:
                print("[WARNING] Cannot close gripper - gripper not initialized")

    def get_state(self):
        return 1 if self.current_state == 'close' else 0
