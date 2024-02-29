python -u executor.py \
    --task_name pretrain \
    --ckpt_name ett_v0.pth \
    --device 2 \
    --run_name ett_pretrain_initial \
    --feature_wise_mse True \
    --source_filename ETTh1.csv \
    --config_name etth1.json \
    --window 35