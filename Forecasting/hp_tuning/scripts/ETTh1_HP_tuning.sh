DATASET="ETT"
SOURCE_FILE="ETTh1.csv"

PRETRAIN_EPOCHS=1
FINETUNE_EPOCHS=1

DEVICE=$1

ENCODER_DIM=$2
ENCODER_DEPTH=$3
ENCODER_HEADS=$4

DECODER_DIM=$5
DECODER_DEPTH=$6
DECODER_HEADS=$7

OUTPUT_PATH="./outputs/ETTh1_${ENCODER_DIM}_${ENCODER_DEPTH}_${ENCODER_HEADS}_${DECODER_DIM}_${DECODER_DEPTH}_${DECODER_HEADS}/"
PRETRAIN_CKPT_DIR="./pretrain_checkpoints_ETTh1/ckpt_${ENCODER_DIM}_${ENCODER_DEPTH}_${ENCODER_HEADS}_${DECODER_DIM}_${DECODER_DEPTH}_${DECODER_HEADS}/"
FINETUNE_CKPT_DIR="./finetune_checkpoints_ETTh1/ckpt_${ENCODER_DIM}_${ENCODER_DEPTH}_${ENCODER_HEADS}_${DECODER_DIM}_${DECODER_DEPTH}_${DECODER_HEADS}/"

ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"

# PRETRAIN
python -u executor.py \
    --task_name pretrain \
    --device $DEVICE \
    --root_path $ROOT_PATH \
    --run_name "pretrain_${SOURCE_FILE}_mask_50" \
    --source_filename $SOURCE_FILE \
    --dataset $DATASET \
    --max_epochs $PRETRAIN_EPOCHS \
    --mask_ratio 0.50 \
    --lr 0.001 \
    --batch_size 16 \
    --encoder_depth $ENCODER_DEPTH \
    --decoder_depth $DECODER_DEPTH \
    --encoder_num_heads $ENCODER_HEADS \
    --encoder_embed_dim $ENCODER_DIM \
    --decoder_num_heads $DECODER_HEADS \
    --decoder_embed_dim $DECODER_DIM \
    --project_name ett \
    --pretrain_checkpoints_dir $PRETRAIN_CKPT_DIR

# FINETUNE WITH NON-FROZEN ENCODER
for pred_len in 96 192 336 720; do
    python -u executor.py \
        --task_name finetune \
        --device $DEVICE \
        --root_path $ROOT_PATH \
        --run_name "finetune_${SOURCE_FILE}_PRED_${pred_len}" \
        --pretrain_run_name "pretrain_${SOURCE_FILE}_mask_50" \
        --freeze_encoder "False" \
        --max_epochs $FINETUNE_EPOCHS \
        --dataset $DATASET \
        --pred_len $pred_len \
        --source_filename $SOURCE_FILE \
        --pretrain_ckpt_name ckpt_best.pth \
        --encoder_depth $ENCODER_DEPTH \
        --encoder_num_heads $ENCODER_HEADS \
        --encoder_embed_dim $ENCODER_DIM \
        --lr 0.0001 \
        --dropout 0.2 \
        --batch_size 16 \
        --project_name ett \
        --output_path $OUTPUT_PATH \
        --pretrain_checkpoints_dir $PRETRAIN_CKPT_DIR \
        --finetune_checkpoints_dir $FINETUNE_CKPT_DIR
done