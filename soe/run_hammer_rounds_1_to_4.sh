#!/usr/bin/env bash
set -e

# source "$HOME/mihomo/proxy.sh"
# proxy 17890

# Run from the repository root in dp3-env, inside a tmux window.
seed=${1:?Usage: bash soe/run_hammer_rounds_1_to_4.sh SEED [GPU]}
gpu=${2:-$seed}
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$gpu"

task=adroit_hammer
root="data/soe/$task/seed_$seed"

checkpoint=data/upload/offline/adroit_hammer/checkpoints/latest.ckpt
demo=data_local/adroit/adroit_hammer_medium.zarr

for round in 1 2 3 4; do
    if [[ "$round" != 1 || "${REUSE_ROUND1_DATA:-0}" != 1 ]]; then
        python soe/collect_soe_rollouts.py \
            --task "$task" --seed "$seed" --round "$round" \
            --checkpoint "$checkpoint" \
            --num-episodes 250 --n-envs 25 --exploration-alpha 2.0 \
            --device "cuda:$gpu" --output "$root/round_$round/rollouts"

        python soe/build_soe_success_dataset.py \
            --task "$task" --seed "$seed" --through-round "$round" \
            --rollout-root "$root" --demo-path "$demo"
    fi

    python train.py --config-name=train_soe_workspace \
        "task=$task" "soe.task=$task" \
        "soe.run_dir=$root/round_$round" "soe.round=$round" "training.seed=$seed" "training.device=cuda:$gpu" \
        policy.vib_latent_dim=16 training.max_grad_steps=50000 training.resume=false \
        logging.project=adroit_offline_debug \
        logging.mode=offline
    checkpoint="$root/round_$round/train/checkpoints/step_050000.ckpt"
done
