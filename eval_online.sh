export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python eval_sum_real.py \
    -c data/outputs/2026.04.28/12.06.34_train_online_vib_real_workspace_wallet/checkpoints/latest.ckpt \
    -o data/eval/wallet/test \
    -d cuda:0 \
    -t 24 \
    -n 20 \
    -m 1000 \
    -s 10