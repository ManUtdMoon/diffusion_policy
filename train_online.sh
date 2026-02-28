export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


python train.py \
    --config-name train_online_vib_real_workspace \
    online_task.base_ckpt="data/outputs/2026.02.27/19.12.52_train_flow_match_vib_unet_image_juicing_s1/checkpoints/latest.ckpt" \
    logging.mode="offline" \
    training.resume_from=null