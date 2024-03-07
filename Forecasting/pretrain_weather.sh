python -u executor.py \
    --task_name pretrain \
    --ckpt_name weather_v0.pth \
    --device 7 \
    --run_name weather_pretrain_initial \
    --source_filename Weather \
    --dataset Weather \
    --max_epochs 2 \
    --encoder_embed_dim 128 \
    --project_name weather