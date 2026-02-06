import time
import numpy as np
from franka.franka_interpolation_controller import FrankaInterface
from pyrobotiqgripper import RobotiqGripper
import joints

def main():
    # Setup connection to Franka
    # Using default IP from FrankaWrapper
    franka = FrankaInterface(ip='172.16.0.1')

    # Setup connection to Gripper
    gripper = RobotiqGripper(portname='/dev/ttyUSB0')
    gripper.goTo(100)
    # gripper.activate() 

    # Wait for initialization (FrankaWrapper moves to joints_init in background)
    time.sleep(1.0)

    print("Moving to pre_grasp...")
    franka.move_to_joint_positions(np.array(joints.PRE_GRASP_JOINTS), time_to_go=2.0)
    time.sleep(2.1) # Wait for motion to complete

    input("Place the spatula and press Enter to continue...")

    print("Moving to grasp...")
    franka.move_to_joint_positions(np.array(joints.GRASP_JOINTS), time_to_go=2.0)
    time.sleep(2.1) # Wait for motion to complete

    print("Closing gripper...")
    gripper.close()
    time.sleep(0.5) # Wait for gripper

    print("Lifting...")
    franka.move_to_joint_positions(np.array(joints.LIFT_JOINTS), time_to_go=3.0)
    time.sleep(3.1)

    print("Moving to start...")
    franka.move_to_joint_positions(np.array(joints.START_JOINTS), time_to_go=2.0)
    time.sleep(2.1)

    print("Done!")
    franka.close()

if __name__ == "__main__":
    main()
