#!/usr/bin/env bash
set -euo pipefail

# Run after: mamba activate zprl

CHECKPOINT="data/outputs/2026.04.25/11.20.28_train_online_vib_robomimic_workspace_square_image/checkpoints/step_500000.ckpt"
SERVER_ADDR="tcp://127.0.0.1:5555"
GUIDANCE="5.0"
SIGMA="0.5"

# linear zeros exp
# 0 2 4 6
for SCHEDULE in exp; do
  for DELAY in 2 4 6; do
    python eval_delay_sum_remote.py \
      -c "${CHECKPOINT}" \
      -o "data/eval/square/rtc/${SCHEDULE}-sigma_${SIGMA}/delay_${DELAY}/" \
      --server_addr "${SERVER_ADDR}" \
      -l "${DELAY}" \
      -s "${SCHEDULE}" \
      -g "${GUIDANCE}" \
      --sigma "${SIGMA}" \
      --rtc
  done
done
