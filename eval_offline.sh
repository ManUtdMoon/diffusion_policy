export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


python eval_real.py \
    -c data/outputs/2026.02.05/19.54.08_train_flow_match_vib_unet_image_flip/checkpoints/latest.ckpt \
    -o eval/flip/rel_999\
    -d cuda:0 \
    -t 16

# python eval_real.py \
#     -c data/outputs/2026.02.05/21.57.51_train_flow_match_vib_unet_image_flip/checkpoints/latest.ckpt \
#     -o eval/flip/abs_max\
#     -d cuda:0 \
#     -t 10