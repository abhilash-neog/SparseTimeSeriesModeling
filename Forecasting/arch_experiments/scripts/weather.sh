DATASET="weather"
SOURCE_FILE="weather"
DEVICE=$1
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

OUTPUT_PATH="./outputs/weather/"

# PRETRAIN
python -u executor.py \
    --task_name pretrain \
    --device $DEVICE \
    --run_name "pretrain_${SOURCE_FILE}" \
    --source_filename $SOURCE_FILE \
    --dataset $DATASET \
    --max_epochs $PRETRAIN_EPOCHS \
    --mask_ratio 0.50 \
    --lr 0.001 \
    --batch_size 16 \
    --encoder_depth 2 \
    --encoder_num_heads 8 \
    --encoder_embed_dim 64 \
    --project_name weather \

# FINETUNE WITH NON-FROZEN ENCODER
for pred_len in 96 192 336 720; do
    python -u executor.py \
        --task_name finetune \
        --device $DEVICE \
        --run_name "finetune_${SOURCE_FILE}_PRED_${pred_len}" \
        --pretrain_run_name "pretrain_${SOURCE_FILE}" \
        --freeze_encoder "False" \
        --max_epochs $FINETUNE_EPOCHS \
        --dataset $DATASET \
        --pred_len $pred_len \
        --source_filename $SOURCE_FILE \
        --pretrain_ckpt_name ckpt_best.pth \
        --encoder_depth 2 \
        --encoder_num_heads 8 \
        --encoder_embed_dim 64 \
        --lr 0.0001 \
        --dropout 0.4 \
        --batch_size 16 \
        --accum_iter 1 \
        --project_name weather \
        --output_path $OUTPUT_PATH
done