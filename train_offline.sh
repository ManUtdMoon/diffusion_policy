export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python train.py \
    --config-name train_flow_match_vib_unet_image_workspace \
    exp_name=n100
