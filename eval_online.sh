export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python eval_sum_real.py \
    -c data/outputs/juicing_s1_v2_resrl/2026.02.26/11.59.34_train_online_res_real_workspace_juicing_s1/checkpoints/latest.ckpt \
    -o data/eval/juicing_s1/play \
    -d cuda:0 \
    -t 10 \
    -n 15 \
    -m 400 \
    -s 2