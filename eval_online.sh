export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python eval_sum_real.py \
    -c data/outputs/box_zprl/2026.03.09/14.23.44_train_online_vib_real_workspace_box/checkpoints/latest.ckpt \
    -o data/eval/box/robust \
    -d cuda:0 \
    -t 16 \
    -n 10 \
    -m 600 \
    -s 2