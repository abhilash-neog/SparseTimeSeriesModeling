#!/bin/bash
#SBATCH -J traffictesting
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q #dgx_normal_q #a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32
#SBATCH --time=18:00:00 # 10 hours
#SBATCH --gres=gpu:1

module reset
module load Anaconda3/2020.11
conda init
# source ~/.bashrc
# conda activate env
source activate env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/env/lib

DATASET="traffic"
SOURCE_FILE="traffic"

PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

DEVICE=$1
TRIAL=$2

ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/traffic"
OUTPUT_PATH="/projects/ml4science/time_series/outputs/traffic/"

FINETUNE_CHECKPOINTS_DIR="/projects/ml4science/time_series/finetune_checkpoints/"
PRETRAIN_CHECKPOINTS_DIR="/projects/ml4science/time_series/pretrain_checkpoints/"

# PRETRAIN
python -u executor.py \
    --task_name pretrain \
    --device $DEVICE \
    --root_path $ROOT_PATH \
    --run_name "v${TRIAL}_pretrain_${SOURCE_FILE}" \
    --source_filename $SOURCE_FILE \
    --dataset $DATASET \
    --max_epochs $PRETRAIN_EPOCHS \
    --mask_ratio 0.50 \
    --lr 0.001 \
    --batch_size 32 \
    --encoder_depth 3 \
    --encoder_num_heads 16 \
    --encoder_embed_dim 128 \
    --decoder_embed_dim 256 \
    --decoder_num_heads 32 \
    --project_name traffic \
    --dropout 0.2 \
    --pretrain_checkpoints_dir $PRETRAIN_CHECKPOINTS_DIR

# FINETUNE WITH NON-FROZEN ENCODER
for pred_len in 96 192 336 720; do
    python -u executor.py \
        --task_name finetune \
        --device $DEVICE \
        --root_path $ROOT_PATH \
        --run_name "v${TRIAL}_finetune_${SOURCE_FILE}_PRED_${pred_len}" \
        --pretrain_run_name "v${TRIAL}_pretrain_${SOURCE_FILE}" \
        --freeze_encoder "False" \
        --max_epochs $FINETUNE_EPOCHS \
        --dataset $DATASET \
        --pred_len $pred_len \
        --source_filename $SOURCE_FILE \
        --pretrain_ckpt_name ckpt_best.pth \
        --encoder_depth 3 \
        --encoder_num_heads 16 \
        --encoder_embed_dim 128 \
        --lr 0.0001 \
        --dropout 0.2 \
        --batch_size 32 \
        --accum_iter 1 \
        --project_name traffic \
        --pretrain_checkpoints_dir $PRETRAIN_CHECKPOINTS_DIR \
        --finetune_checkpoints_dir $FINETUNE_CHECKPOINTS_DIR
done