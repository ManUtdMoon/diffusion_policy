export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python train.py \
    --config-name train_flow_match_vib_unet_image_workspace \
    exp_name=n100-abs_max-H20_To3 \
    task.mode=abs \
    task.dataset.quantile=max \
    n_obs_steps=3 \
    policy.vib_latent_dim=32 \
    policy.vib_beta=0.0001

python train.py \
    --config-name train_flow_match_vib_unet_image_workspace \
    exp_name=n100-rel_max-H20_To3 \
    task.mode=rel \
    task.dataset.quantile=max \
    n_obs_steps=3 \
    policy.vib_latent_dim=32 \
    policy.vib_beta=0.0001

python train.py \
    --config-name train_flow_match_vib_unet_image_workspace \
    exp_name=n100-abs_max-H20 \
    task.mode=abs \
    task.dataset.quantile=max

python train.py \
    --config-name train_flow_match_vib_unet_image_workspace \
    exp_name=n100-rel_max-H20 \
    task.mode=rel \
    task.dataset.quantile=max