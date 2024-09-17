DATASET="ETT"
SOURCE_FILE="ETTh1.csv"

PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

DEVICE=$1

ENCODER_DIM=$2
ENCODER_DEPTH=$3
ENCODER_HEADS=$4

DECODER_DIM=$5
DECODER_DEPTH=$6
DECODER_HEADS=$7

DROPOUT=$8
FC_DROPOUT=$9

OUTPUT_PATH="./outputs/ETTh1_96_${ENCODER_DIM}_${ENCODER_DEPTH}_${ENCODER_HEADS}_${DECODER_DIM}_${DECODER_DEPTH}_${DECODER_HEADS}_dropout_${DROPOUT}_${FC_DROPOUT}/"
PRETRAIN_CKPT_DIR="./pretrain_checkpoints_ETTh1/ckpt_${ENCODER_DIM}_${ENCODER_DEPTH}_${ENCODER_HEADS}_${DECODER_DIM}_${DECODER_DEPTH}_${DECODER_HEADS}_dropout_${DROPOUT}_${FC_DROPOUT}/"
FINETUNE_CKPT_DIR="./finetune_checkpoints_ETTh1/ckpt_${ENCODER_DIM}_${ENCODER_DEPTH}_${ENCODER_HEADS}_${DECODER_DIM}_${DECODER_DEPTH}_${DECODER_HEADS}_dropout_${DROPOUT}_${FC_DROPOUT}/"

ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"

# PRETRAIN
python -u executor.py \
    --task_name pretrain \
    --device $DEVICE \
    --root_path $ROOT_PATH \
    --run_name "pretrain_96_${SOURCE_FILE}_mask_50" \
    --source_filename $SOURCE_FILE \
    --dataset $DATASET \
    --max_epochs $PRETRAIN_EPOCHS \
    --mask_ratio 0.50 \
    --lr 0.001 \
    --batch_size 16 \
    --seq_len 96 \
    --encoder_depth $ENCODER_DEPTH \
    --decoder_depth $DECODER_DEPTH \
    --encoder_num_heads $ENCODER_HEADS \
    --encoder_embed_dim $ENCODER_DIM \
    --decoder_num_heads $DECODER_HEADS \
    --decoder_embed_dim $DECODER_DIM \
    --project_name ett \
    --pretrain_checkpoints_dir $PRETRAIN_CKPT_DIR \
    --dropout $DROPOUT

# FINETUNE WITH NON-FROZEN ENCODER
for pred_len in 96 192 336 720; do
    python -u executor.py \
        --task_name finetune \
        --device $DEVICE \
        --root_path $ROOT_PATH \
        --run_name "finetune_96_${SOURCE_FILE}_PRED_${pred_len}" \
        --pretrain_run_name "pretrain_96_${SOURCE_FILE}_mask_50" \
        --freeze_encoder "False" \
        --max_epochs $FINETUNE_EPOCHS \
        --dataset $DATASET \
        --pred_len $pred_len \
        --seq_len 96 \
        --source_filename $SOURCE_FILE \
        --pretrain_ckpt_name ckpt_best.pth \
        --encoder_depth $ENCODER_DEPTH \
        --encoder_num_heads $ENCODER_HEADS \
        --encoder_embed_dim $ENCODER_DIM \
        --lr 0.0001 \
        --fc_dropout $FC_DROPOUT \
        --batch_size 16 \
        --project_name ett \
        --output_path $OUTPUT_PATH \
        --pretrain_checkpoints_dir $PRETRAIN_CKPT_DIR \
        --finetune_checkpoints_dir $FINETUNE_CKPT_DIR
done