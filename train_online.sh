export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


python train.py \
    --config-name train_online_vib_real_workspace \
    online_task=wallet \
    online_task.base_ckpt="data/outputs/2026.04.25/19.31.07_train_flow_match_vib_unet_image_ddp_wallet/checkpoints/latest.ckpt" \
    logging.mode="offline" \
    training.resume_from="data/outputs/2026.04.28/12.06.34_train_online_vib_real_workspace_wallet/checkpoints/latest.ckpt" \
    training.res_scale=0.15
