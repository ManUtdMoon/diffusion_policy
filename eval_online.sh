export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python eval_sum_real.py \
    -c data/outputs/2026.02.28/15.44.50_train_online_vib_real_workspace_juicing_s1/checkpoints/latest.ckpt \
    -o data/eval/juicing_s1/robust \
    -d cuda:0 \
    -t 16 \
    -n 10 \
    -m 400 \
    -s 2