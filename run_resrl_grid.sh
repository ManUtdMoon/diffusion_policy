#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/run_resrl_grid.sh"
SESSION="${SESSION:-resrl-grid}"
GPUS=(0 1 2 3)
SEEDS=(40 50 60)
TARGET_CLIPS=('[0,1]')

CRITIC_ACTION_INPUTS=(exec)
BASE_CKPTS=(
    "data/upload/offline/tool_hang/cov_rel/checkpoints/epoch_0150.ckpt"
    "data/upload/offline/tool_hang/cov_rel/checkpoints/epoch_0200.ckpt"
    "data/upload/offline/tool_hang/cov_rel/checkpoints/epoch_1000.ckpt"
)

run_worker() {
    source "$HOME/mihomo/proxy.sh"
    proxy 17890

    local gpu="$1"
    local worker_id="$2"
    local run_root="$3"
    local job_idx=0
    local failures=0

    cd "${ROOT_DIR}"
    set +u
    eval "$(mamba shell hook --shell bash)"
    mamba activate zprl
    set -u
    for critic_action_input in "${CRITIC_ACTION_INPUTS[@]}"; do
        for target_clip in "${TARGET_CLIPS[@]}"; do
            for base_ckpt in "${BASE_CKPTS[@]}"; do
                local ckpt_name
                ckpt_name="$(basename "${base_ckpt}" .ckpt)"
                for seed in "${SEEDS[@]}"; do
                    if (( job_idx % ${#GPUS[@]} != worker_id )); then
                        ((job_idx += 1))
                        continue
                    fi

                    local exp_name="resrl:cov_rel-${ckpt_name}-select"
                    local job_dir="${run_root}/${ckpt_name}/seed_${seed}"
                    mkdir -p "${job_dir}"
                    printf '%s\tSTART\tgpu=%s\ttarget_clip=%s\tseed=%s\tcritic_action_input=%s\tbase_ckpt=%s\n' \
                        "$(date --iso-8601=seconds)" "${gpu}" "${target_clip}" "${seed}" "${critic_action_input}" "${base_ckpt}" \
                        >> "${run_root}/status_gpu${gpu}.tsv"

                    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${gpu}" \
                        python train.py \
                        --config-name=train_online_resrl_selection_workspace \
                        online_task=tool_hang_image_abs \
                        "online_task.base_ckpt=${base_ckpt}" \
                        "training.device=cuda:${gpu}" \
                        "training.seed=${seed}" \
                        training.num_steps=125000 \
                        training.buffer_size=125000 \
                        online_task.subtask.enabled=False \
                        online_task.subtask.reward_mode=sparse \
                        n_action_steps=8 \
                        "res_policy.target_clip=${target_clip}" \
                        "res_policy.critic_action_input=${critic_action_input}" \
                        training.utd=4 training.policy_freq=2 \
                        training.n_step=1 training.learning_start=5000 training.training_freq=200 \
                        "exp_name=${exp_name}" \
                        "hydra.run.dir=${job_dir}" \
                        2>&1 | tee "${job_dir}/console.log"
                    local train_status=${PIPESTATUS[0]}

                    if (( train_status == 0 )); then
                        printf '%s\tDONE\tgpu=%s\ttarget_clip=%s\tseed=%s\tcritic_action_input=%s\tbase_ckpt=%s\n' \
                            "$(date --iso-8601=seconds)" "${gpu}" "${target_clip}" "${seed}" "${critic_action_input}" "${base_ckpt}" \
                            >> "${run_root}/status_gpu${gpu}.tsv"
                    else
                        printf '%s\tFAILED(%s)\tgpu=%s\ttarget_clip=%s\tseed=%s\tcritic_action_input=%s\tbase_ckpt=%s\n' \
                            "$(date --iso-8601=seconds)" "${train_status}" "${gpu}" \
                            "${target_clip}" "${seed}" "${critic_action_input}" "${base_ckpt}" \
                            >> "${run_root}/status_gpu${gpu}.tsv"
                        ((failures += 1))
                    fi
                    ((job_idx += 1))
                done
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
    "$((${#TARGET_CLIPS[@]} * ${#SEEDS[@]} * ${#CRITIC_ACTION_INPUTS[@]} * ${#BASE_CKPTS[@]}))" "${SESSION}"
printf 'Run root: %s\n' "${RUN_ROOT}"
printf 'Attach: tmux attach -t %s\n' "${SESSION}"
