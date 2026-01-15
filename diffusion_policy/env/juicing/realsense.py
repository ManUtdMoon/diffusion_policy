import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
from dt_apriltags import Detector



camera_intrinsics = (1346.89, 1346.89, 962.28, 559.75)
# modify after calibrating extrinsics 
X_root_camera = np.array([
    [-0.28586096,  0.63197252, -0.72034314,  0.7757766 ],
    [ 0.95784787,  0.16610085, -0.23438848,  0.13477495],
    [-0.02847747, -0.75698166, -0.65281528,  0.493779  ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])


class RealSense(object):
    def __init__(
            self,
            color_width=960,
            color_height=540,
            fps=30,
            enable_depth=True,
            depth_width=320,
            depth_height=240,
            apriltag_families="tagStandard41h12",
        ):
        self.enable_depth = enable_depth
        self.depth_width = depth_width
        self.depth_height = depth_height

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)

        self.detector = Detector(families=apriltag_families)

    def start(self):
        profile = self.pipeline.start(self.config)

        # get intrinsics
        self.intrinsics = camera_intrinsics
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = (color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy)
        if self.enable_depth:
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_frame = frames.get_depth_frame()
            depth_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
            self.depth_intrinsics = (depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy)
            print(f"Depth Intrinsics: {self.depth_intrinsics}")

    def stop(self):
        self.pipeline.stop()

    def get_frame(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            # frames = self.align.process(frames)

            timestamp = frames.get_timestamp() / 1000  # ms -> s
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame and depth_frame:
                break

        color_image = np.array(color_frame.get_data())
        depth_image = np.array(depth_frame.get_data())

        return {
            'timestamp': timestamp,
            'color': color_image,
            'depth': depth_image,
            'depth_scale': self.depth_scale,
        }
    
    def detect_apriltag(self, color, tag_size=0.03 * 5 / 9, tag_num=3):
        detections = self.detector.detect(
            cv2.cvtColor(color, cv2.COLOR_BGR2GRAY),
            estimate_tag_pose=True,
            camera_params=self.intrinsics,
            tag_size=tag_size
        )
        # print(f'{len(detections)} tags detected.')

        tag_poses = [None] * tag_num
        for detection in detections:
            if detection.tag_id >= tag_num:
                continue
            X_camera_tag = np.eye(4)
            X_camera_tag[:3, :3] = detection.pose_R
            X_camera_tag[:3, 3] = detection.pose_t.flatten()
            X_root_tag = X_root_camera @ X_camera_tag
            tag_poses[detection.tag_id] = X_root_tag
        
        return tag_poses
