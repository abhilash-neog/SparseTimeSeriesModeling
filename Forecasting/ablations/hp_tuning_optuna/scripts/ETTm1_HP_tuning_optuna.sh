DATASET="ETT"
SOURCE_FILE="ETTm1.csv"

PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

DEVICE=$1
db=$2
STATS=$3
STUDY_NAME=$4

pred_len=720
OUTPUT_PATH="./outputs/ETTm1/"
PRETRAIN_CKPT_DIR="./pretrain_checkpoints_ETTm1/"
FINETUNE_CKPT_DIR="./finetune_checkpoints_ETTm1/"

ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"

# PRETRAIN
python -u executor.py \
    --task_name pretrain \
    --device $DEVICE \
    --root_path $ROOT_PATH \
    --pretrain_run_name "pretrain_${SOURCE_FILE}_mask_50" \
    --finetune_run_name "finetune_${SOURCE_FILE}_PRED_${pred_len}"\
    --source_filename $SOURCE_FILE \
    --freeze_encoder "False" \
    --dataset $DATASET \
    --pred_len $pred_len \
    --pretrain_epochs $PRETRAIN_EPOCHS \
    --finetune_epochs $FINETUNE_EPOCHS \
    --mask_ratio 0.50 \
    --lr 0.001 \
    --batch_size 16 \
    --project_name ett \
    --pretrain_checkpoints_dir $PRETRAIN_CKPT_DIR \
    --db $db \
    --stats_file $STATS \
    --study_name $STUDY_NAME