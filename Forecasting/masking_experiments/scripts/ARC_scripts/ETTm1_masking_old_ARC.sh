#!/bin/bash
#SBATCH -J trafficData #optional
#SBATCH --account=ml4science
#SBATCH --partition=a100_normal_q #dgx_normal_q #a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 
#SBATCH --time=0-8:00:00 # 12 hours
#SBATCH --gres=gpu:1

module reset
module load Anaconda3/2020.11
source activate env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/env/lib

DATASET="ETTm1"
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

BASE_PATH="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/ETTm1/"
ROOT_PATHS=$1
TRIAL=$2
MASKINGTYPE=$3
PRED_LEN_LIST=$4
DEVICE=0

SOURCE_FILE="v${TRIAL}_${MASKINGTYPE}_ettm1.csv"

GT_SOURCE_FILE="ETTm1.csv"
GT_ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/ETT/"

OUTPUT_PATH="/projects/ml4science/time_series/outputs_old_params/ettm1/${MASKINGTYPE}/"

PRETRAIN_CHECKPOINTS_DIR="/projects/ml4science/time_series/pretrain_checkpoints_old/"
FINETUNE_CHECKPOINTS_DIR="/projects/ml4science/time_series/finetune_checkpoints_old/"

IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

for id in $ROOT_PATHS; do
    
    root_path="${BASE_PATH}${id}"
    # PRETRAIN
    python -u executor.py \
        --task_name pretrain \
        --device $DEVICE \
        --root_path $root_path \
        --run_name "v${TRIAL}_${MASKINGTYPE}_old_pretrain_${DATASET}_${id}" \
        --source_filename $SOURCE_FILE \
        --dataset $DATASET \
        --max_epochs $PRETRAIN_EPOCHS \
        --mask_ratio 0.50 \
        --lr 0.001 \
        --batch_size 32 \
        --encoder_depth 3 \
        --encoder_num_heads 8 \
        --encoder_embed_dim 32 \
        --decoder_embed_dim 32 \
        --decoder_depth 2 \
        --decoder_num_heads 8 \
        --dropout 0.1 \
        --project_name ett_masking \
        --pretrain_checkpoints_dir $PRETRAIN_CHECKPOINTS_DIR \

    # FINETUNE WITH NON-FROZEN ENCODER
    for pred_len in ${PRED_LEN_ARRAY[@]}; do
        python -u executor.py \
            --task_name finetune \
            --device $DEVICE \
            --root_path $root_path\
            --gt_root_path $GT_ROOT_PATH \
            --gt_source_filename $GT_SOURCE_FILE \
            --run_name "v${TRIAL}_${MASKINGTYPE}_old_finetune_${DATASET}_PRED_${pred_len}_${id}" \
            --pretrain_run_name "v${TRIAL}_${MASKINGTYPE}_old_pretrain_${DATASET}_${id}" \
            --freeze_encoder "False" \
            --max_epochs $FINETUNE_EPOCHS \
            --dataset $DATASET \
            --pred_len $pred_len \
            --source_filename $SOURCE_FILE \
            --pretrain_ckpt_name ckpt_best.pth \
            --encoder_depth 3 \
            --encoder_num_heads 8 \
            --encoder_embed_dim 32 \
            --lr 0.0001 \
            --dropout 0.1 \
            --fc_dropout 0.05 \
            --batch_size 32 \
            --project_name ett_masking \
            --finetune_checkpoints_dir $FINETUNE_CHECKPOINTS_DIR \
            --pretrain_checkpoints_dir $PRETRAIN_CHECKPOINTS_DIR \
            --output_path $OUTPUT_PATH
    done
done