import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
from dt_apriltags import Detector


class RealSense(object):
    def __init__(
            self,
            color_width=960,
            color_height=540,
            fps=30,
            apriltag_families="tagStandard41h12",
        ):

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)

        self.detector = Detector(families=apriltag_families)

    def start(self):
        profile = self.pipeline.start(self.config)

        # get intrinsics
        self.intrinsics = None
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = (color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy)


    def stop(self):
        self.pipeline.stop()

    def get_frame(self):
        while True:
            frames = self.pipeline.wait_for_frames()

            timestamp = frames.get_timestamp() / 1000  # ms -> s
            color_frame = frames.get_color_frame()

            if color_frame:
                break

        color_image = np.array(color_frame.get_data())

        return {
            'timestamp': timestamp,
            'color': color_image,
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
            X_camera_tag = np.eye(4)
            X_camera_tag[:3, :3] = detection.pose_R
            X_camera_tag[:3, 3] = detection.pose_t.flatten()
            X_root_tag = X_root_camera @ X_camera_tag
            tag_poses[detection.tag_id] = X_root_tag
        
        return tag_poses


if __name__ == "__main__":
    rs = RealSense(320, 240)
    rs.start()

    # use cv2 to live stream rs image
    while True:
        frame = rs.get_frame()
        color = frame['color']
        cv2.imshow("RealSense", color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rs.stop()
    cv2.destroyAllWindows()