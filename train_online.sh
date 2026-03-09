export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


python train.py \
    --config-name train_online_vib_real_workspace \
    online_task.base_ckpt="data/outputs/2026.03.05/12.02.32_train_flow_match_vib_unet_image_box/checkpoints/latest.ckpt" \
    logging.mode="offline" \
    training.resume_from="data/outputs/2026.03.09/14.20.08_train_online_vib_real_workspace_box/checkpoints/latest.ckpt" \
    training.res_scale=0.175