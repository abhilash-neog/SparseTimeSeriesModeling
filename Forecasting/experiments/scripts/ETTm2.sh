DATASET="ETT"
SOURCE_FILE="ETTm2"
DEVICE=$1
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

OUTPUT_PATH="./outputs/ettm2/"

# PRETRAIN
python -u executor.py \
    --task_name pretrain \
    --device $DEVICE \
    --run_name "pretrain_${SOURCE_FILE}_mask_50" \
    --source_filename $SOURCE_FILE \
    --dataset $DATASET \
    --max_epochs $PRETRAIN_EPOCHS \
    --mask_ratio 0.50 \
    --lr 0.001 \
    --batch_size 16 \
    --encoder_depth 3 \
    --encoder_num_heads 8 \
    --encoder_embed_dim 8 \
    --project_name ett

# FINETUNE WITH NON-FROZEN ENCODER
for pred_len in 96 192 336 720; do
    python -u executor.py \
    --task_name finetune \
    --device $DEVICE \
    --run_name "finetune_${SOURCE_FILE}_PRED_${pred_len}" \
    --pretrain_run_name "pretrain_${SOURCE_FILE}_mask_50" \
    --freeze_encoder "False" \
    --max_epochs $FINETUNE_EPOCHS \
    --dataset $DATASET \
    --pred_len $pred_len \
    --source_filename $SOURCE_FILE \
    --pretrain_ckpt_name ckpt_best.pth \
    --encoder_depth 3 \
    --encoder_num_heads 8 \
    --encoder_embed_dim 8 \
    --lr 0.0001 \
    --dropout 0.0 \
    --batch_size 64 \
    --project_name ett \
    --output_path $OUTPUT_PATH

done