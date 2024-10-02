DATASET="Lake"

PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

DEVICE=$1
db=$2
STATS=$3
STUDY_NAME=$4
TRIALS=$5

pred_len=21
seq_len=21

ROOT_PATH="/raid/sepideh/Project_MissTSM/FCR/"
SOURCE_FILE="FCR_missing"

# PRETRAIN
python -u executor.py \
    --task_name pretrain \
    --device $DEVICE \
    --target 'daily_median_chla_interp_ugL' 'daily_median_watertemp_interp_degC'\
    --root_path $ROOT_PATH \
    --pretrain_run_name "pretrain_${SOURCE_FILE}_mask_50" \
    --finetune_run_name "finetune_${SOURCE_FILE}_PRED_${pred_len}"\
    --source_filename $SOURCE_FILE \
    --freeze_encoder "False" \
    --dataset $DATASET \
    --pred_len $pred_len \
    --seq_len $seq_len \
    --pretrain_epochs $PRETRAIN_EPOCHS \
    --finetune_epochs $FINETUNE_EPOCHS \
    --mask_ratio 0.50 \
    --batch_size 64 \
    --features MD \
    --db $db \
    --stats_file $STATS \
    --study_name $STUDY_NAME \
    --ntrials $TRIALS \
    --pretrain_lr 0.001\
    --finetune_lr 0.0001