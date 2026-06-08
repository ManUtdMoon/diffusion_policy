#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
  --num_processes 4 \
  --multi_gpu \
  --main_process_port 29501 \
  train.py \
  --config-name=train_flow_match_vib_unet_image_accelerate_workspace \
  task=tool_hang_image_abs \
  exp_name=dinov3_weighted-vib-dit-prev_a
