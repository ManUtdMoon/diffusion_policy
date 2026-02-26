export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


python eval_real.py \
    -c data/outputs/2026.02.25/16.58.54_train_flow_match_vib_unet_image_juicing_s1/checkpoints/latest.ckpt \
    -o data/eval/juicing_s1/test\
    -d cuda:0 \
    -t 10

# python eval_real.py \
#     -c data/outputs/2026.02.05/21.57.51_train_flow_match_vib_unet_image_flip/checkpoints/latest.ckpt \
#     -o eval/flip/abs_max\
#     -d cuda:0 \
#     -t 10