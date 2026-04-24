#!/bin/bash

# ============================================================
# Multi-GPU training launcher
# Usage: bash launch.sh
# ============================================================

SESSION="train"
GPUS=(0 1 2)
SEEDS=(10 20 30)
EXP_NAME="zprl-utd5-s0.15-no_ln-no_qn"
BASE_CKPT="data/upload/offline/square/checkpoints/epoch_0600-score_0.460.ckpt"

# kill old session with the same name (optional, comment out if not wanted)
tmux kill-session -t $SESSION 2>/dev/null

# create a new detached tmux session
tmux new-session -d -s $SESSION

for i in "${!GPUS[@]}"; do
    GPU=${GPUS[$i]}
    SEED=${SEEDS[$i]}

    CMD="MUJOCO_EGL_DEVICE_ID=${GPU} python train.py \
        --config-name=train_online_vib_robomimic_workspace \
        training.seed=${SEED} \
        exp_name=${EXP_NAME} \
        training.device=cuda:${GPU} \
        online_task.base_ckpt=${BASE_CKPT} \
        training.res_scale=0.15"

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