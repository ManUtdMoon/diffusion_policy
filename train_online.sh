export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


python train.py \
    --config-name train_online_vib_real_workspace \
    online_task.base_ckpt="data/outputs/2026.02.10/20.32.27_train_flow_match_vib_unet_image_flip/checkpoints/latest.ckpt" \
    logging.mode="offline" \
    training.resume_from="data/outputs/2026.02.12/16.16.01_train_online_vib_real_workspace_flip" \
    n_action_steps=16 \
    res_policy.num_qs=5 \
    training.utd=2.0 \
    training.res_scale=0.2