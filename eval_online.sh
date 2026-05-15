export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python eval_sum_real.py \
    -c data/outputs/2026.05.11/15.07.02_train_online_vib_real_workspace_wallet/checkpoints/step-25100.ckpt \
    -o data/eval/wallet/robust-human \
    -d cuda:0 \
    -t 24 \
    -n 10 \
    -m 1200 \
    -s 10