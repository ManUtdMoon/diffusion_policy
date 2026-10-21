#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/run_oneleg_dsrl_grid.sh"
SESSION="${SESSION:-oneleg-dsrl-grid}"
GPUS=(0 1 2 3)
ACTION_STEPS=(8)
CASES=(1)
SEEDS=(42)
NOMINAL_ENV_STEPS=1000000
BASE_CKPT="data/upload/oneleg/checkpoints/epoch_0055-mse_0.020349.ckpt"

run_worker() {
    local gpu="$1"
    local worker_id="$2"
    local run_root="$3"
    local job_idx=0
    local failures=0

    cd "${ROOT_DIR}"
    set +u
    eval "$(mamba shell hook --shell bash)"
    mamba activate oneleg
    set -u
    export CUDA_VISIBLE_DEVICES=1,0,2,3
    export OMP_NUM_THREADS=2
    for ta in "${ACTION_STEPS[@]}"; do
        local decisions=$(( NOMINAL_ENV_STEPS / ta ))
        local utd
        utd="$(awk -v ta="${ta}" 'BEGIN {print ta / 2}')"
        local n_step=$(( 8 / ta ))
        local learning_start=$(( 40000 / ta ))
        local history_lengths=("${CASES[@]}")
        local sub_ratio=1
        if (( ta == 1 )); then
            history_lengths=(1 5)
            sub_ratio=2
        elif (( ta == 2 )); then
            history_lengths=(1 3)
        fi
        for history_length in "${history_lengths[@]}"; do
            for seed in "${SEEDS[@]}"; do
                if (( job_idx % ${#GPUS[@]} != worker_id )); then
                    ((job_idx += 1))
                    continue
                fi

                local exp_name="oneleg_dsrl_h${history_length}s${sub_ratio}_ta${ta}-n${n_step}"
                local job_dir="${run_root}/ta_${ta}/history_${history_length}/seed_${seed}"
                mkdir -p "${job_dir}"
                printf '%s\tSTART\tgpu=%s\thistory_length=%s\tseed=%s\tta=%s\n' \
                    "$(date --iso-8601=seconds)" "${gpu}" "${history_length}" "${seed}" "${ta}" \
                    >> "${run_root}/status_gpu${gpu}.tsv"

                python train.py \
                    --config-name=train_online_reactive_history_noise_oneleg_workspace \
                    online_task=oneleg_image \
                    "online_task.base_ckpt=${BASE_CKPT}" \
                    "training.device=cuda:${gpu}" \
                    "training.seed=${seed}" \
                    "training.num_steps=${decisions}" \
                    "training.buffer_size=${decisions}" \
                    "n_action_steps=${ta}" n_noise_steps=1 \
                    "noise_policy.history_length=${history_length}" \
                    "noise_policy.sub_ratio=${sub_ratio}" \
                    "training.utd=${utd}" \
                    "training.n_step=${n_step}" \
                    "training.learning_start=${learning_start}" \
                    "exp_name=${exp_name}" \
                    "logging.name=${exp_name}_seed${seed}" \
                    "hydra.run.dir=${job_dir}" \
                    2>&1 | tee "${job_dir}/console.log"
                local train_status=${PIPESTATUS[0]}

                if (( train_status == 0 )); then
                    printf '%s\tDONE\tgpu=%s\thistory_length=%s\tseed=%s\tta=%s\n' \
                        "$(date --iso-8601=seconds)" "${gpu}" "${history_length}" "${seed}" "${ta}" \
                        >> "${run_root}/status_gpu${gpu}.tsv"
                else
                    printf '%s\tFAILED(%s)\tgpu=%s\thistory_length=%s\tseed=%s\tta=%s\n' \
                        "$(date --iso-8601=seconds)" "${train_status}" "${gpu}" \
                        "${history_length}" "${seed}" "${ta}" \
                        >> "${run_root}/status_gpu${gpu}.tsv"
                    ((failures += 1))
                fi
                ((job_idx += 1))
            done
        done
    done

    printf 'GPU %s finished with %s failed job(s).\n' "${gpu}" "${failures}"
    return "${failures}"
}

if [[ "${1:-}" == "--worker" ]]; then
    run_worker "$2" "$3" "$4"
    exit $?
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required."
    exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}"
    exit 1
fi

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/data/outputs/oneleg_dsrl_grid_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_ROOT}"

for worker_id in "${!GPUS[@]}"; do
    gpu="${GPUS[$worker_id]}"
    window="gpu${gpu}"
    worker_cmd="bash '${SCRIPT_PATH}' --worker '${gpu}' '${worker_id}' '${RUN_ROOT}'; exec zsh"
    if (( worker_id == 0 )); then
        tmux new-session -d -s "${SESSION}" -n "${window}" "${worker_cmd}"
    else
        tmux new-window -d -t "${SESSION}" -n "${window}" "${worker_cmd}"
    fi
done

printf 'Started %s jobs in tmux session %s.\n' \
    "$((${#CASES[@]} * ${#SEEDS[@]} * ${#ACTION_STEPS[@]}))" "${SESSION}"
printf 'Run root: %s\n' "${RUN_ROOT}"
printf 'Attach: tmux attach -t %s\n' "${SESSION}"
