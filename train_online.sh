export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


python train.py \
    --config-name train_online_vib_real_workspace \
    online_task.base_ckpt="data/outputs/2026.02.09/17.08.36_train_flow_match_vib_unet_image_flip/checkpoints/latest.ckpt" \
    logging.mode="offline" \
    training.resume_from="data/outputs/2026.02.10/17.17.53_train_online_vib_real_workspace_flip"