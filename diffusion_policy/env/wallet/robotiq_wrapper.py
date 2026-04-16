import time
import threading
from pyrobotiqgripper import RobotiqGripper

GRIPPER_OPEN_POSITION = 96

class RobotiqWrapper:
    def __init__(self, robot):
        portname='/dev/ttyUSB0' if robot == 'franka' else '/dev/ttyUSB1'
        
        # 检查设备是否存在
        import os
        if not os.path.exists(portname):
            print(f"[WARNING] Robotiq gripper port {portname} does not exist!")
            self.gripper = None
            self.current_state = 'open'
            self.current_position = 0
            self.change_state = False
            return
        
        try:
            self.gripper = RobotiqGripper(portname=portname)
            # self.gripper.reset()
            print(f"[RobotiqWrapper] Activating gripper on {portname}...")
            self.gripper.activate()
            print(f"[RobotiqWrapper] Gripper activated successfully")
            
            self.current_state = 'open'
            self.current_position = GRIPPER_OPEN_POSITION
            self.change_state = True
            
            self.gripper_thread = threading.Thread(target=self._monitor_gripper)
            self.gripper_thread.daemon = True
            self.gripper_thread.start()
            
        except Exception as e:
            print(f"[WARNING] Failed to initialize Robotiq gripper: {e}")
            self.gripper = None
            self.current_state = 'open'
            self.current_position = 0
            self.change_state = False

    def _monitor_gripper(self):
        while True:
            if self.gripper is not None and self.change_state:
                try:
                    if self.current_state == 'position':
                        self.gripper.goTo(self.current_position)
                    elif self.current_state == 'open':
                        self.gripper.goTo(GRIPPER_OPEN_POSITION)
                    elif self.current_state == 'close':
                        self.gripper.close()
                    self.change_state = False
                except Exception as e:
                    print(f"[WARNING] Gripper control error: {e}")
            time.sleep(1 / 30)

    def open(self):
        if self.current_state != 'open':
            self.current_state = 'open'
            self.current_position = GRIPPER_OPEN_POSITION
            self.change_state = True
    
    def close(self):
        if self.current_state != 'close':
            self.current_state = 'close'
            self.current_position = 255
            self.change_state = True

    def set_position(self, position):
        clipped_position = max(0, min(255, int(position)))
        if abs(self.current_position - clipped_position) > 1 or self.current_state != 'position':
            self.current_position = clipped_position
            self.current_state = 'position'
            self.change_state = True
        if self.gripper is None:
            print("[WARNING] Cannot set gripper position - gripper not initialized")

    def get_state(self):
        return self.current_position / 255.0
