#!/usr/bin/env bash

set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
eval "$(mamba shell hook --shell bash)"
mamba activate oneleg
export CUDA_VISIBLE_DEVICES=1,0,2,3

accelerate launch \
    --num_processes 2 \
    --multi_gpu \
    --main_process_port "${MAIN_PROCESS_PORT:-29501}" \
    train.py \
    --config-name=train_flow_match_oneleg_image_workspace \
    exp_name=default \
    "$@"
