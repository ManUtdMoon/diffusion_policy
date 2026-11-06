#!/usr/bin/env bash
set -e

# source "$HOME/mihomo/proxy.sh"
# proxy 17890

# Run from the repository root in dp3-env, inside a tmux window.
seed=${1:?Usage: bash soe/run_box_close_rounds_1_to_4.sh SEED [GPU]}
gpu=${2:-$seed}
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$gpu"

task=metaworld_box-close
root="data/soe/$task/seed_$seed"

checkpoint=data/upload/offline/box-close/new.ckpt
demo=data_local/metaworld/metaworld_box-close_medium-expert.zarr

for round in 1 2 3 4; do
    python soe/collect_soe_rollouts.py \
        --task "$task" --seed "$seed" --round "$round" \
        --checkpoint "$checkpoint" \
        --num-episodes 250 --n-envs 25 --exploration-alpha 2.0 \
        --device "cuda:$gpu" --output "$root/round_$round/rollouts"

    python soe/build_soe_success_dataset.py \
        --task "$task" --seed "$seed" --through-round "$round" \
        --rollout-root "$root" --demo-path "$demo"

    python train.py --config-name=train_soe_workspace \
        "task=$task" "soe.task=$task" \
        "soe.run_dir=$root/round_$round" "soe.round=$round" "training.seed=$seed" "training.device=cuda:$gpu" \
        policy.vib_latent_dim=8 training.max_grad_steps=50000 training.resume=false \
        dataloader.batch_size=256 dataloader.num_workers=12 \
        logging.project=metaworld_offline_debug \
        logging.mode=offline
    checkpoint="$root/round_$round/train/checkpoints/step_050000.ckpt"
done
