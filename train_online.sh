export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


python train.py \
    --config-name train_online_vib_real_workspace \
    online_task=wallet \
    online_task.base_ckpt="data/outputs/2026.04.20/15.30.42_train_flow_match_vib_unet_image_ddp_wallet/checkpoints/latest.ckpt" \
    logging.mode="offline" \
    training.resume_from=null \
    training.res_scale=0.15
