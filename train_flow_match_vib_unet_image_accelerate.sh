#!/bin/bash
set -euo pipefail

exec zsh -lc '
source ~/.zshrc
mamba activate zprl

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NUM_PROCESSES=$(python - <<"INNER_PY"
import os
visible = [x.strip() for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
print(len(visible))
INNER_PY
)
else
  NUM_PROCESSES=${NUM_PROCESSES:-2}
fi

MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}
CONFIG_NAME=${CONFIG_NAME:-train_flow_match_vib_unet_image_accelerate_workspace}

accelerate launch   --num_processes ${NUM_PROCESSES}   --multi_gpu   --main_process_port ${MAIN_PROCESS_PORT}   train.py   --config-name=${CONFIG_NAME}   "$@"
' zsh "$@"
