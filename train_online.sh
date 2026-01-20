export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


python train.py \
    --config-name train_online_vib_real_workspace \
    online_task.base_ckpt="data/outputs/2026.01.14/11.26.23_train_flow_match_vib_unet_image_juicing_s1/checkpoints/latest.ckpt" \
    logging.mode="offline" \
    training.resume_from="data/outputs/2026.01.19/14.24.01_train_online_vib_real_workspace_juicing_s1"