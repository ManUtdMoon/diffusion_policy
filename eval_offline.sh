export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


python eval_real.py \
    -c data/outputs/2026.04.20/15.30.42_train_flow_match_vib_unet_image_ddp_wallet/checkpoints/latest.ckpt \
    -o data/eval/wallet/test_3rd-both_tcp-Ta24 \
    -d cuda:0 \
    -t 24 \
    -s 10
