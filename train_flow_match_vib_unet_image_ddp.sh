#!/usr/bin/env zsh

set -euo pipefail

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

SCRIPT_DIR="${0:A:h}"
cd "${SCRIPT_DIR}"

if [[ "${1:-}" == "--help-script" ]]; then
    cat <<'EOF'
Usage:
  ./train_flow_match_vib_unet_image_ddp.sh [hydra overrides ...]

Examples:
  ./train_flow_match_vib_unet_image_ddp.sh task=wallet training.seed=1
  ./train_flow_match_vib_unet_image_ddp.sh dataloader.batch_size=128 logging.mode=offline
  CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 ./train_flow_match_vib_unet_image_ddp.sh task=wallet

Launcher env:
  NUM_PROCESSES       Number of processes / GPUs to launch. Default: inferred from CUDA_VISIBLE_DEVICES, else 1.
  MAIN_PROCESS_PORT   Main process port. Default: 29500.
EOF
    exit 0
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_list=(${(s:,:)CUDA_VISIBLE_DEVICES})
    inferred_num_processes="${#gpu_list[@]}"
else
    inferred_num_processes=1
fi

num_processes="${NUM_PROCESSES:-$inferred_num_processes}"
main_process_port="${MAIN_PROCESS_PORT:-29500}"

launch_args=(
    --num_processes "${num_processes}"
    --main_process_port "${main_process_port}"
)

if (( num_processes > 1 )); then
    launch_args=(--multi_gpu "${launch_args[@]}")
fi

python -m accelerate.commands.launch \
    "${launch_args[@]}" \
    train.py \
    --config-name train_flow_match_vib_unet_image_ddp_workspace \
    "$@"
