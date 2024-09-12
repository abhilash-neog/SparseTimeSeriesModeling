export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/env/lib

DATASET="ETT"
SOURCE_FILE="ETTh1"

PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

DEVICE=0
ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/"

OUTPUT_PATH="/projects/ml4science/time_series/outputs/etth1/"

FINETUNE_CHECKPOINTS_DIR="/projects/ml4science/time_series/finetune_checkpoints/"
PRETRAIN_CHECKPOINTS_DIR="/projects/ml4science/time_series/pretrain_checkpoints/"

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
    --encoder_depth 3 \
    --encoder_num_heads 16 \
    --encoder_embed_dim 32 \
    --project_name ett \
    --pretrain_checkpoints_dir $PRETRAIN_CHECKPOINTS_DIR

# FINETUNE
for pred_len in 96 192 336 720; do
    python -u executor.py \
        --task_name finetune \
        --device $DEVICE \
        --root_path $ROOT_PATH \
        --run_name "finetune_${SOURCE_FILE}_PRED_${pred_len}" \
        --pretrain_run_name "pretrain_${SOURCE_FILE}_mask_50" \
        --freeze_encoder "True" \
        --max_epochs $FINETUNE_EPOCHS \
        --dataset $DATASET \
        --pred_len $pred_len \
        --source_filename $SOURCE_FILE \
        --pretrain_ckpt_name ckpt_best.pth \
        --encoder_depth 3 \
        --encoder_num_heads 16 \
        --encoder_embed_dim 32 \
        --lr 0.0001 \
        --dropout 0.2 \
        --batch_size 16 \
        --project_name ett \
        --output_path $OUTPUT_PATH \
        --finetune_checkpoints_dir $FINETUNE_CHECKPOINTS_DIR \
        --pretrain_checkpoints_dir $PRETRAIN_CHECKPOINTS_DIR \
done