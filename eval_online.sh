export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python eval_sum_real.py \
    -c data/outputs/2026.02.12/14.27.47_train_online_vib_real_workspace_flip-8500/checkpoints/latest.ckpt \
    -o data/eval/flip/robust_size \
    -d cuda:0 \
    -t 16 \
    -n 15 \
    -m 250 \
    -s 2