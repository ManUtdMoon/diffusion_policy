#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
  --num_processes 2 \
  --multi_gpu \
  --main_process_port 29501 \
  train.py \
  --config-name=train_flow_match_vib_unet_image_accelerate_workspace \
  task=square_image_abs \
  exp_name=conti_t-ddp
