if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)


import shutil
from pathlib import Path

import click
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

from diffusion_policy.env.flip.franka.common.pose_util import pose_to_mat
from diffusion_policy.env.flip.franka.franka_interpolation_controller import tx_tip_flange


camera_intrinsics = (
    304.62615966796875,
    304.2835388183594,
    162.17880249023438,
    124.3314208984375,
)  # flip, 320x240
X_root_camera = np.array([
    [0.9986473, 0.0423569, 0.0301571, 0.5986900],
    [0.0118751, -0.7504600, 0.6608092, -0.3329134],
    [0.0506215, -0.6595572, -0.7499479, 0.3920208],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)  # flip main


def _wrist_to_ee_position(tip_pose):
    flange_mat = pose_to_mat(tip_pose) @ tx_tip_flange
    p_root_flange = flange_mat[:3, 3]
    rot = flange_mat[:3, :3]
    p_flange_ee = np.array([0.0, 0.21, 0.33], dtype=np.float64)
    return p_root_flange + rot @ p_flange_ee


def _project_points(points_root, in_camera_frame):
    if points_root.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)

    if in_camera_frame:
        points_cam = points_root
    else:
        X_camera_root = np.linalg.inv(X_root_camera)
        points_h = np.concatenate(
            [points_root, np.ones((points_root.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        points_cam = (X_camera_root @ points_h.T).T[:, :3]

    fx, fy, cx, cy = camera_intrinsics
    z = points_cam[:, 2]
    valid = z > 1e-6
    u = fx * (points_cam[:, 0] / z) + cx
    v = fy * (points_cam[:, 1] / z) + cy
    return np.stack([u, v], axis=1), valid


def _draw_points(bgr, pixels, valid, colors_bgr, radius):
    h, w = bgr.shape[:2]
    for i in range(pixels.shape[0]):
        if not valid[i]:
            continue
        u = int(round(pixels[i, 0]))
        v = int(round(pixels[i, 1]))
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(bgr, (u, v), radius, colors_bgr[i], -1)


def _reds_bgr(total):
    if total <= 0:
        return []
    if total == 1:
        rgb = plt.cm.Reds(0.8)[:3]
        return [(int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))]

    colors = []
    for i in range(total):
        t = 0.15 + 0.75 * (i / float(total - 1))
        rgb = plt.cm.Reds(t)[:3]  # reds for zprl, blues for resrl
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
    return colors


def _save_frames(frames, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, rgb_hwc in enumerate(frames):
        cv2.imwrite(str(output_dir / f"frame_{i:04d}.png"), rgb_hwc[:, :, ::-1])


def _save_mp4(frames, output_path, fps):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")
    for rgb_hwc in frames:
        writer.write(rgb_hwc[:, :, ::-1])
    writer.release()


@click.command()
@click.option("--traj-h5", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--fps", type=int, default=10, show_default=True)
@click.option("--save-mode", type=click.Choice(["video", "frames"]), default="video", show_default=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option("--in-camera-frame", is_flag=True)
@click.option("--radius", type=int, default=4, show_default=True)
def main(traj_h5, fps, save_mode, output_path, in_camera_frame, radius):
    with h5py.File(traj_h5, "r") as f:
        dense_images = f["dense_raw/image"][:]  # (T_dense, 3, H, W), RGB CHW
        exec_chunk = f["sparse/chunk_action"][:]  # (N_chunk, Ta, Da)

    if exec_chunk.ndim != 3:
        raise ValueError(f"Expected sparse/chunk_action to have shape (N, Ta, Da), got {exec_chunk.shape}")

    n_chunk, chunk_size, action_dim = exec_chunk.shape
    if action_dim < 6:
        raise ValueError(f"Action dim must be at least 6, got {action_dim}")

    n_steps = min(dense_images.shape[0] - 1, n_chunk * chunk_size)
    if n_steps <= 0:
        raise ValueError("No frames to render.")

    proj_chunks = []
    for c in range(n_chunk):
        xyz = np.stack(
            [_wrist_to_ee_position(exec_chunk[c, i, :6].astype(np.float64)) for i in range(chunk_size)],
            axis=0,
        )
        pixels, valid = _project_points(xyz, in_camera_frame)
        proj_chunks.append((pixels, valid))

    chunk_colors = _reds_bgr(chunk_size)
    frames = []
    for t in range(n_steps):
        c = t // chunk_size
        local_i = t % chunk_size
        bgr = np.moveaxis(dense_images[t + 1], 0, -1)[:, :, ::-1].copy()

        pixels, valid = proj_chunks[c]
        _draw_points(bgr, pixels, valid, chunk_colors, radius)

        if valid[local_i]:
            u = int(round(pixels[local_i, 0]))
            v = int(round(pixels[local_i, 1]))
            if 0 <= u < bgr.shape[1] and 0 <= v < bgr.shape[0]:
                cv2.circle(bgr, (u, v), radius + 3, (0, 255, 0), -1)

        frames.append(bgr[:, :, ::-1])

    if save_mode == "frames":
        if output_path is None:
            output_path = traj_h5.parent / "flip_traj_frames"
        if output_path.exists():
            shutil.rmtree(output_path)
        _save_frames(frames, output_path)
        print(f"Saved frames dir: {output_path}")
    else:
        if output_path is None:
            output_path = traj_h5.parent / "flip_traj.mp4"
        if output_path.is_dir():
            raise ValueError(f"video mode expects file output path, got directory: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _save_mp4(frames, output_path, fps)
        print(f"Saved local video: {output_path}")

    print(f"Rendered frames: {n_steps}")


if __name__ == "__main__":
    main()
