if __name__ == "__main__":
    import sys
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)


import cv2
import h5py
import numpy as np
import click
from pathlib import Path
import wandb
import shutil


camera_intrinsics = (676.3255004882812, 676.2794189453125, 488.832275390625, 272.2892150878906)
X_root_camera = np.array([
    [-0.28586096, 0.63197252, -0.72034314, 0.7757766],
    [0.95784787, 0.16610085, -0.23438848, 0.13477495],
    [-0.02847747, -0.75698166, -0.65281528, 0.493779],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def _rpy_to_matrix(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def _wrist_to_ee_position(wrist_pose):
    x, y, z, roll, pitch, yaw = wrist_pose.tolist()
    R = _rpy_to_matrix(roll, pitch, yaw)
    p_root_wrist = np.array([x, y, z], dtype=np.float64)
    p_wrist_ee = np.array([0.0, 0.0, 0.1], dtype=np.float64)
    return p_root_wrist + R @ p_wrist_ee


def _project_points(points_root, in_camera_frame):
    if points_root.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    if in_camera_frame:
        points_cam = points_root
    else:
        X_camera_root = np.linalg.inv(X_root_camera)
        points_h = np.concatenate([points_root, np.ones((points_root.shape[0], 1), dtype=np.float64)], axis=1)
        points_cam = (X_camera_root @ points_h.T).T[:, :3]

    fx, fy, cx, cy = camera_intrinsics
    z = points_cam[:, 2]
    valid = z > 1e-6
    u = fx * (points_cam[:, 0] / z) + cx
    v = fy * (points_cam[:, 1] / z) + cy
    return np.stack([u, v], axis=1), valid


def _draw_points(bgr, pixels, valid, color_bgr, radius):
    h, w = bgr.shape[:2]
    for i in range(pixels.shape[0]):
        if not valid[i]:
            continue
        u = int(round(pixels[i, 0]))
        v = int(round(pixels[i, 1]))
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(bgr, (u, v), radius, color_bgr, -1)


def _blue_shade_bgr(idx, total):
    # Keep in blue family (BGR): blue channel fixed high, G/R vary by index.
    if total <= 1:
        return (255, 80, 40)
    t = idx / float(total - 1)  # 0 -> 1
    g = int(40 + 140 * t)       # 40..180
    r = int(20 + 90 * t)        # 20..110
    return (255, g, r)


def _save_frames(frames, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, rgb_hwc in enumerate(frames):
        bgr_hwc = rgb_hwc[:, :, ::-1]
        cv2.imwrite(str(output_dir / f"frame_{i:04d}.png"), bgr_hwc)


def _save_mp4(frames, output_path, fps):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")
    for rgb_hwc in frames:
        writer.write(rgb_hwc[:, :, ::-1])
    writer.release()


@click.command()
@click.option("--traj-h5", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--pred-root", type=str, default="pred/random_sum", show_default=True)
@click.option("--fps", type=int, default=30, show_default=True)
@click.option("--save-mode", type=click.Choice(["video", "frames"]), default="video", show_default=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None)
@click.option("--wandb-project", type=str, default="juicing-visualize", show_default=True)
@click.option("--wandb-run-name", type=str, default=None)
@click.option("--wandb-mode", type=click.Choice(["online", "offline", "disabled"]), default="offline", show_default=True)
@click.option("--in-camera-frame", is_flag=True)
def main(traj_h5, pred_root, fps, save_mode, output_path, wandb_project, wandb_run_name, wandb_mode, in_camera_frame):
    with h5py.File(traj_h5, "r") as f:
        dense_images = f["dense_raw/image"][:]  # (T_dense, 3, H, W)
        exec_chunk = f["sparse/chunk_action"][:]  # (N_chunk, Ta, Da)
        pred_chunk = f[f"{pred_root}/chunk_action"][:]  # (N_chunk, 9, Ta, Da)

    n_chunk_exec, chunk_size, _ = exec_chunk.shape
    n_chunk_pred, n_pred, chunk_size_pred, _ = pred_chunk.shape
    if chunk_size_pred != chunk_size:
        raise ValueError(f"chunk size mismatch: exec={chunk_size}, pred={chunk_size_pred}")
    n_chunk = min(n_chunk_exec, n_chunk_pred)

    n_steps = min(dense_images.shape[0] - 1, n_chunk * chunk_size)
    if n_steps <= 0:
        raise ValueError("No frames to render.")

    exec_proj = []
    pred_proj = []
    for c in range(n_chunk):
        exec_xyz = np.stack([_wrist_to_ee_position(exec_chunk[c, i, :6].astype(np.float64)) for i in range(chunk_size)], axis=0)
        exec_pixels, exec_valid = _project_points(exec_xyz, in_camera_frame)
        exec_proj.append((exec_pixels, exec_valid))

        curr_pred = []
        for k in range(n_pred):
            pred_xyz = np.stack([_wrist_to_ee_position(pred_chunk[c, k, i, :6].astype(np.float64)) for i in range(chunk_size)], axis=0)
            pred_pixels, pred_valid = _project_points(pred_xyz, in_camera_frame)
            curr_pred.append((pred_pixels, pred_valid))
        pred_proj.append(curr_pred)

    frames = []
    for t in range(n_steps):
        c = t // chunk_size
        local_i = t % chunk_size
        bgr = np.moveaxis(dense_images[t + 1], 0, -1)[:, :, ::-1].copy()

        # Predicted chunks: blue
        for k in range(n_pred):
            pixels, valid = pred_proj[c][k]
            _draw_points(bgr, pixels, valid, color_bgr=_blue_shade_bgr(k, n_pred), radius=2)

        # Executed chunk: red
        exec_pixels, exec_valid = exec_proj[c]
        _draw_points(bgr, exec_pixels, exec_valid, color_bgr=(0, 0, 255), radius=4)
        if exec_valid[local_i]:
            u = int(round(exec_pixels[local_i, 0]))
            v = int(round(exec_pixels[local_i, 1]))
            if 0 <= u < bgr.shape[1] and 0 <= v < bgr.shape[0]:
                cv2.circle(bgr, (u, v), 7, (0, 255, 0), -1)

        frames.append(bgr[:, :, ::-1])
    if save_mode == "frames":
        if output_path is None:
            output_path = traj_h5.parent / "traj_projection_with_pred_frames"
        if output_path.exists():
            shutil.rmtree(output_path)
        _save_frames(frames, output_path)
    else:
        if output_path is None:
            output_path = traj_h5.parent / "traj_projection_with_pred.mp4"
        if output_path.is_dir():
            raise ValueError(f"video mode expects file output path, got directory: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _save_mp4(frames, output_path, fps)

    run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        mode=wandb_mode,
        config={
            "traj_h5": str(traj_h5),
            "pred_root": pred_root,
            "fps": fps,
            "n_steps_rendered": int(n_steps),
            "n_pred_chunks": int(n_pred),
            "output_type": save_mode,
        }
    )
    if save_mode == "video":
        run.log({"projection_video": wandb.Video(str(output_path), fps=fps, format="mp4")})
    else:
        run.log({"projection_video": wandb.Image(str(output_path / "frame_0000.png"))})
    run.finish()

    if save_mode == "video":
        print(f"Saved local video: {output_path}")
    else:
        print(f"Saved frames dir: {output_path}")
    print(f"Rendered frames: {n_steps}")


if __name__ == "__main__":
    main()
