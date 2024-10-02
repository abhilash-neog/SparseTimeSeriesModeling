DATASET="weather"
SOURCE_FILE="weather"

PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

DEVICE=$1
db=$2
STATS=$3
STUDY_NAME=$4
TRIALS=$5

pred_len=720
OUTPUT_PATH="./outputs/weather_new/"
PRETRAIN_CKPT_DIR="/raid/abhilash/optuna_pretrain_checkpoints_weather_new/"
FINETUNE_CKPT_DIR="/raid/abhilash/optuna_finetune_checkpoints_weather_new/"

ROOT_PATH="/raid/abhilash/forecasting_datasets/weather/"

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
    --batch_size 16 \
    --accum_iter 1 \
    --project_name weather \
    --pretrain_checkpoints_dir $PRETRAIN_CKPT_DIR \
    --db $db \
    --stats_file $STATS \
    --study_name $STUDY_NAME \
    --ntrials $TRIALS \
    --pretrain_lr 0.001\
    --finetune_lr 0.0001