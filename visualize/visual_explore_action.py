if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)


import numpy as np
import click
import yaml
import json
from pathlib import Path
import matplotlib.pyplot as plt

from diffusion_policy.env.juicing.realsense import RealSense
from diffusion_policy.env.flip.franka.common.pose_util import *
from diffusion_policy.env.flip.franka.franka_interpolation_controller import tx_flange_tip, tx_tip_flange


camera_intrinsics = (1346.89, 1346.89, 962.28, 559.75)
camera_intrinsics = (1352.6510009765625, 1352.558837890625, 977.66455078125, 544.5784301757812)
camera_intrinsics = (1370.8177490234375, 1369.2760009765625, 969.8046264648438, 559.4913940429688) # flip, 1080p
camera_intrinsics = (304.62615966796875, 304.2835388183594, 162.17880249023438, 124.3314208984375) # flip, 240p
# modify after calibrating extrinsics 
X_root_camera = np.array([
    [-0.28586096,  0.63197252, -0.72034314,  0.7757766 ],
    [ 0.95784787,  0.16610085, -0.23438848,  0.13477495],
    [-0.02847747, -0.75698166, -0.65281528,  0.493779  ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
X_root_camera = np.array([
    [ 0.9986473,   0.0423569,   0.0301571,   0.598690  ],
    [ 0.0118751,  -0.7504600,   0.6608092,  -0.3329134 ],
    [ 0.0506215,  -0.6595572,  -0.7499479,   0.3920208 ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
]) # flip main


# franka
def _wrist_to_ee_position(tip_pose):
    # tip_pose is xyz + rotvec
    # 1. transfer to flange pose
    flange_mat = pose_to_mat(tip_pose) @ tx_tip_flange
    p_root_flange = flange_mat[:3, 3]
    R = flange_mat[:3, :3]
    p_flange_ee = np.array([0.0, 0.21, 0.33], dtype=np.float64)

    # 2. from flange to spatula
    p_root_ee = p_root_flange + R @ p_flange_ee
    return p_root_ee


def _load_xyz_points(actions_path, action_key, ratio):
    points = []
    with open(actions_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            action = entry.get(action_key)
            if action is None:
                for fallback_key in ("sum_action", "policy_action", "action", "base_action"):
                    if fallback_key in entry:
                        action = entry[fallback_key]
                        break
            if action is None:
                continue
            arr = np.array(action, dtype=np.float64)
            if arr.ndim == 1:
                points.append(_wrist_to_ee_position(arr[:6]))
            elif arr.ndim == 2:
                for i in range(arr.shape[0]):
                    points.append(_wrist_to_ee_position(arr[i, :6]))
            break
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if ratio is None or ratio <= 1:
        return np.stack(points, axis=0)
    return np.stack(points[::ratio], axis=0)


def _project_points(points_root, intrinsics, X_root_camera, in_camera_frame):
    if points_root.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    if in_camera_frame:
        points_cam = points_root
    else:
        X_camera_root = np.linalg.inv(X_root_camera)
        ones = np.ones((points_root.shape[0], 1), dtype=np.float64)
        points_h = np.concatenate([points_root, ones], axis=1)
        points_cam = (X_camera_root @ points_h.T).T[:, :3]
    fx, fy, cx, cy = intrinsics
    z = points_cam[:, 2]
    valid = z > 1e-6
    u = fx * (points_cam[:, 0] / z) + cx
    v = fy * (points_cam[:, 1] / z) + cy
    return np.stack([u, v], axis=1), valid


def _color_for_idx(i, n):
    if n <= 1:
        return (0.12, 0.47, 0.71)
    t = i / (n - 1)
    return plt.cm.Reds(t)[:3]


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing episode_actions.txt"
)
@click.option(
    "--action-key",
    type=str,
    default="sum_action",
    show_default=True,
    help="Which action key to visualize from each entry"
)
@click.option("--ratio", type=int, default=10, show_default=True)
@click.option(
    "--image",
    "image_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional background image to draw on"
)
@click.option("--width", type=int, default=1920, show_default=True)
@click.option("--height", type=int, default=1080, show_default=True)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output image path"
)
@click.option(
    "--in-camera-frame",
    is_flag=True,
    help="Treat action xyz as already in camera coordinates"
)
def main(input_dir, action_key, ratio, image_path, width, height, output_path, in_camera_frame):
    actions_path = input_dir / "episode_actions.txt"
    if not actions_path.exists():
        raise FileNotFoundError(f"Missing {actions_path}")

    points_root = _load_xyz_points(actions_path, action_key, ratio)
    pixels, valid = _project_points(points_root, camera_intrinsics, X_root_camera, in_camera_frame)

    print(points_root.shape)

    if image_path is not None:
        img = plt.imread(str(image_path))
        height, width = img.shape[:2]
    else:
        img = np.zeros((height, width, 3), dtype=np.uint8)

    if output_path is None:
        output_path = input_dir / "trajectory_pixels.png"

    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(img)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    for idx, (uv, ok) in enumerate(zip(pixels, valid)):
        if not ok:
            continue
        u, v = uv[0], uv[1]
        if u < 0 or u >= width or v < 0 or v >= height:
            continue
        color = _color_for_idx(idx, len(pixels))
        ax.plot(u, v, marker="o", markersize=15, color=color)

    fig.savefig(str(output_path))
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
