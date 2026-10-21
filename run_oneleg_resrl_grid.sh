#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/run_oneleg_resrl_grid.sh"
SESSION="${SESSION:-oneleg-resrl-grid}"
GPUS=(0 1 2 3)
SCALES=(0.1 0.15 0.2)
SEEDS=(40 50)
HLEN=(1 2 4)
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
    for scale in "${SCALES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for hlen in "${HLEN[@]}"; do
                if (( job_idx % ${#GPUS[@]} != worker_id )); then
                    ((job_idx += 1))
                    continue
                fi

                local exp_name="resrl_h${hlen}_a8r2_s${scale}-qbr_3e-4"
                local job_dir="${run_root}/scale_${scale}/seed_${seed}"
                mkdir -p "${job_dir}"
                printf '%s\tSTART\tgpu=%s\tscale=%s\tseed=%s\thlen=%s\n' \
                    "$(date --iso-8601=seconds)" "${gpu}" "${scale}" "${seed}" "${hlen}" \
                    >> "${run_root}/status_gpu${gpu}.tsv"

                # 50000 nominal global_step / Tr8 = 6250 effective decisions.
                python train.py \
                    --config-name=train_online_reactive_history_oneleg_workspace \
                    online_task=oneleg_image \
                    "online_task.base_ckpt=${BASE_CKPT}" \
                    "training.device=cuda:${gpu}" \
                    "training.seed=${seed}" \
                    training.num_steps=500000 \
                    training.buffer_size=500000 \
                    n_action_steps=8 n_rl_steps=2 \
                    "res_policy.history_length=${hlen}"\
                    res_policy.sub_ratio=1 \
                    "training.res_scale=${scale}" \
                    "exp_name=${exp_name}" \
                    "hydra.run.dir=${job_dir}" \
                    2>&1 | tee "${job_dir}/console.log"
                local train_status=${PIPESTATUS[0]}

                if (( train_status == 0 )); then
                    printf '%s\tDONE\tgpu=%s\tscale=%s\tseed=%s\thlen=%s\n' \
                        "$(date --iso-8601=seconds)" "${gpu}" "${scale}" "${seed}" "${hlen}" \
                        >> "${run_root}/status_gpu${gpu}.tsv"
                else
                    printf '%s\tFAILED(%s)\tgpu=%s\tscale=%s\tseed=%s\thlen=%s\n' \
                        "$(date --iso-8601=seconds)" "${train_status}" "${gpu}" "${scale}" "${seed}" "${hlen}" \
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

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/data/outputs/oneleg_resrl_grid_$(date +%Y%m%d_%H%M%S)}"
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
    "$((${#SCALES[@]} * ${#SEEDS[@]} * ${#HLEN[@]}))" "${SESSION}"
printf 'Run root: %s\n' "${RUN_ROOT}"
printf 'Attach: tmux attach -t %s\n' "${SESSION}"
