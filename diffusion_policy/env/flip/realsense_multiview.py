import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import pyrealsense2 as rs


# Default multi-view config. Update serial numbers to match your setup.
DEFAULT_MULTI_CAMERAS = OrderedDict({
    "record": {
        "serial_number": "213622078748",
        "color_width": 1920,
        "color_height": 1080,
        "fps": 30,
        "exposure": 750.0,
    },
    "color": {
        "serial_number": "134722070628",
        "color_width": 320,
        "color_height": 240,
        "fps": 30,
    },
})


class RealSenseView:
    """Single-camera wrapper following realsense_flip.py usage pattern."""
    def __init__(self, name, serial_number, color_width=320, color_height=240, fps=30, exposure=None):
        self.name = name
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
        self.exposure = exposure
        self.started = False

    def start(self):
        profile = self.pipeline.start(self.config)
        sensor = profile.get_device().query_sensors()[1]
        profile_color = profile.get_stream(rs.stream.color)
        intr_color = profile_color.as_video_stream_profile().get_intrinsics()
        print(f"Camera {self.name} intrinsics: {intr_color.fx}, {intr_color.fy}, {intr_color.ppx}, {intr_color.ppy}")
        if self.exposure is not None:
            print(f"Setting exposure of camera {self.name} to {self.exposure}")
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, float(self.exposure))
        else:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1)
        self.started = True

    def stop(self):
        if self.started:
            self.pipeline.stop()
            self.started = False

    def get_frame(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                break
        return {
            "timestamp": frames.get_timestamp() / 1000.0,
            "color": np.asarray(color_frame.get_data()),
        }


class MultiViewRealSense:
    """Threaded multi-camera manager, one reader thread per view."""
    def __init__(self, camera_cfg=None):
        self.camera_cfg = camera_cfg if camera_cfg is not None else OrderedDict()
        self.cameras = OrderedDict()
        self.latest = {}
        self.lock = threading.Lock()
        self.threads = []
        self.running = False

    def start(self):
        if len(self.camera_cfg) == 0:
            return
        self.running = True
        for cam_name, kwargs in self.camera_cfg.items():
            cam = RealSenseView(name=cam_name, **kwargs)
            cam.start()
            self.cameras[cam_name] = cam
            self.latest[cam_name] = None
            t = threading.Thread(target=self._reader_loop, args=(cam_name,), daemon=True)
            t.start()
            self.threads.append(t)

    def _reader_loop(self, cam_name):
        cam = self.cameras[cam_name]
        while self.running:
            try:
                frame = cam.get_frame()["color"]
                with self.lock:
                    self.latest[cam_name] = frame
            except Exception:
                # Keep the reader alive for transient camera failures.
                time.sleep(0.01)

    def get_frames(self, timeout_s=2.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self.lock:
                ready = all(self.latest.get(name) is not None for name in self.cameras.keys())
                if ready:
                    return {name: self.latest[name].copy() for name in self.cameras.keys()}
            time.sleep(0.005)
        missing = [name for name in self.cameras.keys() if self.latest.get(name) is None]
        raise RuntimeError(f"Timeout waiting for camera frames. Missing: {missing}")

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=1.0)
        self.threads.clear()
        for cam in self.cameras.values():
            cam.stop()
        self.cameras.clear()
        self.latest.clear()


def test_preview(camera_cfg=None):
    camera = MultiViewRealSense(camera_cfg if camera_cfg is not None else DEFAULT_MULTI_CAMERAS)
    camera.start()
    try:
        while True:
            frames = camera.get_frames()
            tiles = []
            for cam_name, frame in frames.items():
                vis = frame.copy()
                if vis.shape[0] < 720:
                    vis = cv2.resize(vis, (1920, 1080))
                cv2.putText(vis, cam_name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                tiles.append(vis)
            preview = np.concatenate(tiles, axis=1)
            cv2.imshow("multiview_realsense", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_preview()
