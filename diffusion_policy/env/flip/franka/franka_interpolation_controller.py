import os
import time
import enum
import numpy as np
import multiprocessing as mp
import zerorpc
from multiprocessing.managers import SharedMemoryManager
from scipy.spatial.transform import Rotation as R, Slerp
from collections import deque

from .shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from .shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from .common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from .common.precise_sleep import precise_wait
from .common.pose_util import pose_to_mat, mat_to_pose


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    RESET_HOME = 3
    SERVOJ = 4


tx_flangerot90_tip = np.identity(4)
tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])

tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3,:3] = R.from_euler('x', [np.pi/2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3,:3] = R.from_euler('z', [np.pi/4]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)

class FrankaInterface:
    def __init__(self, ip, port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")
        self.prev_pose = None
        self.prev_prev_pose = None

    def get_ee_pose(self):
        flange_pose = np.array(self.server.get_ee_pose())
        tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
        return tip_pose
    
    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())
    
    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.prev_pose = None
        self.prev_prev_pose = None
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(
            Kx.tolist(),
            Kxd.tolist()
        )
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        dt = 1 / 30
        if self.prev_pose is not None and self.prev_prev_pose is not None:
            v_prev = (self.prev_pose[:3] - self.prev_prev_pose[:3]) / dt
            v_now = (pose[:3] - self.prev_pose[:3]) / dt
            a_now = np.linalg.norm((v_now - v_prev) / dt)
            # print(f"[FrankaInterface] Current acceleration: {a_now:.2f} m/s^2")
            if a_now >= 20.0:
                print(f"[FrankaInterface] ERROR: high acceleration {a_now:.2f} m/s^2")
                self.server.terminate_current_policy()
                self.server.close()
                exit()
        self.server.update_desired_ee_pose(pose.tolist())
        self.prev_prev_pose = self.prev_pose
        self.prev_pose = pose

    def terminate_current_policy(self):
        self.server.terminate_current_policy()

    def close(self):
        self.server.close()


class FrankaInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        robot_ip,
        robot_port=4242,
        frequency=1000,
        Kx_scale=1.0,
        Kxd_scale=1.0,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0,
        lookahead_steps=3,
        smoothing_weight=0.7
        ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        lookahead_steps: number of future poses to consider for smoothing
        smoothing_weight: weight for exponential smoothing (0.0-1.0, higher = smoother)
        """

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FrankaPositionalController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.frequency = frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.lookahead_steps = lookahead_steps
        self.smoothing_weight = smoothing_weight
        self.pose_buffer = deque(maxlen=lookahead_steps)

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'target_joints': np.zeros((7,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
            'from_cartesian': False
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd','get_joint_velocities'),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
            
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        # Add new pose to buffer for look-ahead smoothing
        self.pose_buffer.append({'pose': pose, 'duration': duration})
        
        # Compute smoothed pose if we have enough poses in buffer
        if len(self.pose_buffer) >= 2:
            smoothed_pose = self._compute_smoothed_pose()
        else:
            smoothed_pose = pose

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': smoothed_pose, # smoothed_pose, pose
            'duration': duration
        }
        self.input_queue.put(message)
    
    def _compute_smoothed_pose(self):
        """
        Compute smoothed pose using weighted average of poses in buffer
        """
        if len(self.pose_buffer) < 2:
            return self.pose_buffer[-1]['pose']
        
        # Get current and look-ahead poses
        poses = [item['pose'] for item in self.pose_buffer]
        
        # Separate position and rotation
        positions = np.array([p[:3] for p in poses])
        rotations = [R.from_rotvec(p[3:]) for p in poses]
        
        # Apply exponential weighting for positions
        weights = np.array([self.smoothing_weight ** i for i in range(len(poses))])
        weights = weights / weights.sum()
        
        # Smooth positions
        smoothed_pos = np.average(positions, axis=0, weights=weights)
        
        # Smooth rotations using weighted SLERP
        if len(rotations) >= 2:
            # Start with the most recent rotation
            smoothed_rot = rotations[0]
            
            # Sequentially blend with each subsequent rotation using SLERP
            for i in range(1, len(rotations)):
                # Create a Slerp interpolator between current smoothed rotation and next rotation
                times = np.array([0.0, 1.0])
                # Stack rotations into a single Rotation object
                rots = R.from_matrix(np.stack([smoothed_rot.as_matrix(), rotations[i].as_matrix()]))
                slerp = Slerp(times, rots)
                
                # Interpolate based on the relative weight
                weight = weights[i] / weights[:i+1].sum()
                smoothed_rot = slerp(weight)
        else:
            smoothed_rot = rotations[0]
        
        # Combine smoothed position and rotation
        smoothed_pose = np.zeros(6)
        smoothed_pose[:3] = smoothed_pos
        smoothed_pose[3:] = smoothed_rot.as_rotvec()
        
        return smoothed_pose
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def reset_home(self):
        # Clear pose buffer when resetting
        self.pose_buffer.clear()
        msg = {
            'cmd': Command.RESET_HOME.value
        }
        self.input_queue.put(msg)

    def servoJ(self, joints, from_cartesian=True):
        """
        Move robot in joint space.
        If from_cartesian=True, terminate cartesian impedance before moving.
        """
        assert self.is_alive()
        joints = np.array(joints)
        assert joints.shape == (7,)

        self.pose_buffer.clear()
        msg = {
            'cmd': Command.SERVOJ.value,
            'target_joints': joints,
            'from_cartesian': bool(from_cartesian),
        }
        self.input_queue.put(msg)
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
            
        # start polymetis interface
        robot = FrankaInterface(self.robot_ip, self.robot_port)

        try:
            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")
            
            # init pose
            if self.joints_init is not None:
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=self.joints_init_duration
                )

            # main loop
            dt = 1. / self.frequency
            curr_pose = robot.get_ee_pose()

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            # start franka cartesian impedance policy
            robot.start_cartesian_impedance(
                Kx=self.Kx,
                Kxd=self.Kxd
            )
            cartesian_policy_active = True

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                tip_pose = pose_interp(t_now)
                flange_pose = mat_to_pose(pose_to_mat(tip_pose) @ tx_tip_flange)

                # send command to robot
                if cartesian_policy_active:
                    robot.update_desired_ee_pose(flange_pose)

                # update robot state
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(robot, func_name)()

                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    # commands = self.input_queue.get_all()
                    # n_cmd = len(commands['cmd'])
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[FrankaPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESET_HOME.value:
                        if cartesian_policy_active:
                            robot.terminate_current_policy()
                            cartesian_policy_active = False
                        robot.move_to_joint_positions(
                            positions=np.asarray(self.joints_init),
                            time_to_go=self.joints_init_duration
                        )

                        robot.start_cartesian_impedance(self.Kx, self.Kxd)
                        cartesian_policy_active = True

                        curr_pose = robot.get_ee_pose()
                        t_now_reset = time.monotonic()
                        pose_interp = PoseTrajectoryInterpolator(
                            times=[t_now_reset],
                            poses=[curr_pose]
                        )
                        last_waypoint_time = t_now_reset
                        if self.verbose:
                            print("[FrankaPositionalController] Already reset to home")
                    elif cmd == Command.SERVOJ.value:
                        target_joints = np.asarray(command['target_joints'])
                        from_cartesian = bool(command['from_cartesian'])

                        if from_cartesian and cartesian_policy_active:
                            robot.terminate_current_policy()
                            cartesian_policy_active = False
                        robot.move_to_joint_positions(
                            positions=target_joints,
                            time_to_go=None
                        )
                        if self.verbose:
                            print(
                                "[FrankaPositionalController] SERVOJ done: joints={} from_cartesian={}".format(
                                    target_joints, from_cartesian
                                )
                            )
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[FrankaPositionalController] Actual frequency {1/(time.monotonic() - t_now)}")

        finally:
            # manditory cleanup
            # terminate
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            robot.terminate_current_policy()
            del robot
            self.ready_event.set()

            if self.verbose:
                print(f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")
