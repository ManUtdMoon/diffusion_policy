#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/run_resrl_grid.sh"
SESSION="${SESSION:-resrl-grid}"
GPUS=(0 1 2 3)
CASES=(
    "3"
    "5"
    "7"
)
SEEDS=(40 50 60)
CRITIC_ACTION_INPUTS=(base_res)
BASE_CKPT="data/upload/offline/tool_hang/novib/checkpoints/latest.ckpt"

run_worker() {
    local gpu="$1"
    local worker_id="$2"
    local run_root="$3"
    local job_idx=0
    local failures=0

    cd "${ROOT_DIR}"
    eval "$(mamba shell hook --shell bash)"
    mamba activate zprl
    for critic_action_input in "${CRITIC_ACTION_INPUTS[@]}"; do
        for history_length in "${CASES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                if (( job_idx % ${#GPUS[@]} != worker_id )); then
                    ((job_idx += 1))
                    continue
                fi

                local exp_name="resrl:oa_h${history_length}s2_gru-hoa_clipq-qbr-a8r1-buf8"
                local job_dir="${run_root}/critic_${critic_action_input}/a8r1-history_${history_length}/seed_${seed}"
                mkdir -p "${job_dir}"
                printf '%s\tSTART\tgpu=%s\thistory_length=%s\tseed=%s\tcritic_action_input=%s\n' \
                    "$(date --iso-8601=seconds)" "${gpu}" "${history_length}" "${seed}" "${critic_action_input}" \
                    >> "${run_root}/status_gpu${gpu}.tsv"

                MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu}" \
                    python train.py \
                    --config-name=train_online_reactive_history_robomimic_workspace \
                    online_task=tool_hang_image_abs \
                    "online_task.base_ckpt=${BASE_CKPT}" \
                    "training.device=cuda:${gpu}" \
                    "training.seed=${seed}" \
                    training.num_steps=1000000 \
                    training.buffer_size=1000000 \
                    online_task.subtask.enabled=False \
                    online_task.subtask.reward_mode=sparse \
                    n_action_steps=8 n_rl_steps=1 \
                    "res_policy.history_length=${history_length}" \
                    "res_policy.critic_action_input=${critic_action_input}" \
                    training.utd=0.5 training.policy_freq=2 \
                    training.n_step=8 training.learning_start=40000 training.training_freq=1000 \
                    "exp_name=${exp_name}" \
                    "logging.name=${exp_name}_tool_hang_seed${seed}" \
                    "hydra.run.dir=${job_dir}" \
                    2>&1 | tee "${job_dir}/console.log"
                local train_status=${PIPESTATUS[0]}

                if (( train_status == 0 )); then
                    printf '%s\tDONE\tgpu=%s\thistory_length=%s\tseed=%s\tcritic_action_input=%s\n' \
                        "$(date --iso-8601=seconds)" "${gpu}" "${history_length}" "${seed}" "${critic_action_input}" \
                        >> "${run_root}/status_gpu${gpu}.tsv"
                else
                    printf '%s\tFAILED(%s)\tgpu=%s\thistory_length=%s\tseed=%s\tcritic_action_input=%s\n' \
                        "$(date --iso-8601=seconds)" "${train_status}" "${gpu}" \
                        "${history_length}" "${seed}" "${critic_action_input}" \
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

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/data/outputs/resrl_grid_$(date +%Y%m%d_%H%M%S)}"
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
    "$((${#CASES[@]} * ${#SEEDS[@]} * ${#CRITIC_ACTION_INPUTS[@]}))" "${SESSION}"
printf 'Run root: %s\n' "${RUN_ROOT}"
printf 'Attach: tmux attach -t %s\n' "${SESSION}"
