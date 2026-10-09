#!/bin/bash

# ============================================================
# Multi-GPU training launcher
# Usage: bash launch.sh
# ============================================================

SESSION="train"
GPUS=(0 1 2)
SEEDS=(40 50 60)
EXP_NAME="resrl:h2mlp_xavier-u1_2-A4R2-bufN2"
BASE_CKPT="data/upload/offline/square/novib/checkpoints/epoch_0500-score_0.450.ckpt"

# kill old session with the same name (optional, comment out if not wanted)
tmux kill-session -t $SESSION 2>/dev/null

# create a new detached tmux session
tmux new-session -d -s $SESSION

for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    SEED=${SEEDS[$i]}

    CMD="MUJOCO_EGL_DEVICE_ID=${GPU} python train.py \
        --config-name=train_online_reactive_history_robomimic_workspace \
        training.seed=${SEED} \
        exp_name=${EXP_NAME} \
        training.device=cuda:${GPU} \
        online_task.base_ckpt=${BASE_CKPT} \
        online_task=square_image_abs \
        training.num_steps=500000 \
        training.buffer_size=500000 \
        online_task.subtask.enabled=False \
        online_task.subtask.reward_mode=sparse \
        training.utd=1.0"

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
