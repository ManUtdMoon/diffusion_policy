#!/bin/bash
set -e

source "$HOME/mihomo/proxy.sh"
proxy 17890

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
  --num_processes 2 \
  --multi_gpu \
  --main_process_port 29501 \
  train.py \
  --config-name=train_diffusion_image_accelerate_workspace \
  task=tool_hang_image_abs \
  exp_name=unet-cov_rel \
  training.seed=10
