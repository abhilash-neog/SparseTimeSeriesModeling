python -u executor.py \
    --task_name pretrain \
    --ckpt_name wandbTestModel.pth \
    --device 4 \
    --run_name wandbtesting \
    --feature_wise_rmse True \
    --max_epochs 3 \
    --n2one_ft False \
    --project_name 2dmasking \
    --encoder_embed_dim 8