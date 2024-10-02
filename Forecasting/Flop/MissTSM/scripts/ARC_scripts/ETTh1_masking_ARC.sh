#!/bin/bash
#SBATCH -J ecltesting
#SBATCH --account=ml4science
#SBATCH --partition=a100_normal_q #dgx_normal_q #a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --time=30:00:00 # 24 hours
#SBATCH --gres=gpu:1

module reset
module load Anaconda3/2020.11
source activate env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/env/lib

DATASET="ETTh1"
PRETRAIN_EPOCHS=1
FINETUNE_EPOCHS=1

BASE_PATH="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/ETTh1/"
ROOT_PATHS=$1
TRIAL=$2
MASKINGTYPE=$3
DEVICE=0
PRED_LEN_LIST=$4

SOURCE_FILE="v${TRIAL}_${MASKINGTYPE}_etth1"

GT_SOURCE_FILE="ETTh1"
GT_ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/ETT/"

OUTPUT_PATH="/projects/ml4science/time_series/outputs/etth2/${MASKINGTYPE}/"

PRETRAIN_CHECKPOINTS_DIR="/projects/ml4science/time_series/pretrain_checkpoints/"
FINETUNE_CHECKPOINTS_DIR="/projects/ml4science/time_series/finetune_checkpoints/"

IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

for id in $ROOT_PATHS; do
    
    root_path="${BASE_PATH}${id}"
    # PRETRAIN
    python -u executor.py \
        --task_name pretrain \
        --device $DEVICE \
        --root_path $root_path \
        --run_name "v${TRIAL}_${MASKINGTYPE}_pretrain_${DATASET}_${id}" \
        --source_filename $SOURCE_FILE \
        --dataset $DATASET \
        --max_epochs $PRETRAIN_EPOCHS \
        --mask_ratio 0.50 \
        --lr 0.001 \
        --batch_size 16 \
        --encoder_depth 2 \
        --encoder_num_heads 8 \
        --encoder_embed_dim 8 \
        --project_name ett_masking \
        --pretrain_checkpoints_dir $PRETRAIN_CHECKPOINTS_DIR \
        --trial $TRIAL

    # FINETUNE WITH NON-FROZEN ENCODER
    for pred_len in ${PRED_LEN_ARRAY[@]}; do
        python -u executor.py \
            --task_name finetune \
            --device $DEVICE \
            --root_path $root_path\
            --gt_root_path $GT_ROOT_PATH \
            --gt_source_filename $GT_SOURCE_FILE \
            --run_name "v${TRIAL}_${MASKINGTYPE}_finetune_${DATASET}_PRED_${pred_len}_${id}" \
            --pretrain_run_name "v${TRIAL}_${MASKINGTYPE}_pretrain_${DATASET}_${id}" \
            --freeze_encoder "False" \
            --max_epochs $FINETUNE_EPOCHS \
            --dataset $DATASET \
            --pred_len $pred_len \
            --source_filename $SOURCE_FILE \
            --pretrain_ckpt_name ckpt_best.pth \
            --encoder_depth 2 \
            --encoder_num_heads 8 \
            --encoder_embed_dim 8 \
            --lr 0.0001 \
            --dropout 0.4 \
            --batch_size 16 \
            --finetune_checkpoints_dir $FINETUNE_CHECKPOINTS_DIR \
            --pretrain_checkpoints_dir $PRETRAIN_CHECKPOINTS_DIR \
            --output_path $OUTPUT_PATH
    done
done