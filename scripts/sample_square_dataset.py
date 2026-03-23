"""
Sample 1000 random items from square_image_abs dataset and save as a .pt file.
Dataset params match square_image_abs.yaml + train_flow_match_vib_unet_image_workspace.yaml defaults.
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset

# ---- config matching square_image_abs.yaml + workspace defaults ----
DATASET_PATH = "/media/datahub-2/ydj/robomimicv030/square/mh/image_v141_subset_abs.hdf5"
SHAPE_META = {
    "obs": {
        "agentview_image":          {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eef_pos":           {"shape": [3]},
        "robot0_eef_quat":          {"shape": [4]},
        "robot0_gripper_qpos":      {"shape": [2]},
    },
    "action": {"shape": [10]},
}

# workspace yaml: horizon=16, n_obs_steps=1, n_action_steps=4, n_latency_steps=0
HORIZON       = 16
N_OBS_STEPS   = 1
N_ACT_STEPS   = 4
N_LATENCY     = 0
PAD_BEFORE    = N_OBS_STEPS - 1 + N_LATENCY   # = 0
PAD_AFTER     = N_ACT_STEPS - 1               # = 3

SEED          = 10
NUM_SAMPLES   = 1000
OUTPUT_PATH   = f"data/square_image_abs_{NUM_SAMPLES}samples.pt"

# ---- build dataset ----
print("Loading dataset ...")
dataset = RobomimicReplayImageDataset(
    shape_meta=SHAPE_META,
    dataset_path=DATASET_PATH,
    horizon=HORIZON,
    pad_before=PAD_BEFORE,
    pad_after=PAD_AFTER,
    n_obs_steps=N_OBS_STEPS,
    abs_action=True,
    rotation_rep="rotation_6d",
    use_legacy_normalizer=False,
    use_cache=True,
    seed=SEED,
    val_ratio=0.00,
    num_demo=100,
)
print(f"Dataset size: {len(dataset)}")

# ---- random sampling ----
rng = random.Random(SEED)
indices = rng.sample(range(len(dataset)), NUM_SAMPLES)

subset = Subset(dataset, indices)
loader = DataLoader(subset, batch_size=64, num_workers=0, shuffle=False)

# ---- collect batches ----
all_batches = []
print(f"Sampling {NUM_SAMPLES} items ...")
for batch in loader:
    all_batches.append(batch)

# merge along batch dim
def cat_nested(batches):
    keys = batches[0].keys()
    out = {}
    for k in keys:
        v0 = batches[0][k]
        if isinstance(v0, dict):
            out[k] = cat_nested([b[k] for b in batches])
        else:
            out[k] = torch.cat([b[k] for b in batches], dim=0)
    return out

samples = cat_nested(all_batches)

# ---- save ----
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
torch.save(samples, OUTPUT_PATH)
print(f"Saved {NUM_SAMPLES} samples to {OUTPUT_PATH}")

# ---- quick sanity check ----
print("Keys:", list(samples.keys()))
print("obs keys:", list(samples["obs"].keys()))
for k, v in samples["obs"].items():
    print(f"  obs[{k}]: {tuple(v.shape)} {v.dtype}")
print(f"  action: {tuple(samples['action'].shape)} {samples['action'].dtype}")
