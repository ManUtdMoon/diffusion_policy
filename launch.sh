#!/bin/bash

# ============================================================
# Multi-GPU training launcher
# Usage: bash launch.sh
# ============================================================

SESSION="train"
GPUS=(0 1 2)
SEEDS=(10 20 30)
EXP_NAME="rlt-b20-td3_s0.01-no_drop"
BASE_CKPT="data/outputs/2026.06.09/11.41.28_train_diffusion_image_accelerate_tool_hang_image_abs/checkpoints/epoch_0800-score_0.420.ckpt"

# kill old session with the same name (optional, comment out if not wanted)
tmux kill-session -t $SESSION 2>/dev/null

# create a new detached tmux session
tmux new-session -d -s $SESSION

for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    SEED=${SEEDS[$i]}

    CMD="MUJOCO_EGL_DEVICE_ID=${GPU} python train.py \
        --config-name=train_online_direct_robomimic_workspace \
        training.seed=${SEED} \
        exp_name=${EXP_NAME} \
        training.device=cuda:${GPU} \
        online_task.base_ckpt=${BASE_CKPT} \
        online_task=tool_hang_image_abs \
        training.num_steps=200000 \
        direct_policy.bc_loss_weight=20.0 \
        direct_policy.fixed_std=0.01"

    if [ $i -eq 0 ]; then
        tmux send-keys -t $SESSION "mamba activate zprl" Enter
        tmux send-keys -t $SESSION "$CMD" Enter
    else
        sleep 2
        tmux new-window -t $SESSION
        tmux send-keys -t $SESSION "mamba activate zprl" Enter
        tmux send-keys -t $SESSION "$CMD" Enter
    fi
done

# attach to the session so you can see all windows
tmux attach -t $SESSION
