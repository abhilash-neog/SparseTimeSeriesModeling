#!/bin/bash
#SBATCH -J ecl #optional
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --time=30:00:00 # 12 hours
#SBATCH --gres=gpu:1

module reset
module load Anaconda3/2020.11
source activate env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/env/lib

DATASET="electricity"
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

BASE_PATH="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/electricity/"
ROOT_PATHS=$1
DEVICE=0
TRIAL=$2
MASKINGTYPE=$3
PRED_LEN_LIST=$4

SOURCE_FILE="v${TRIAL}_${MASKINGTYPE}_electricity.csv"

GT_SOURCE_FILE="electricity.csv"
GT_ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/electricity/"

OUTPUT_PATH="/projects/ml4science/time_series/outputs_upd/${MASKINGTYPE}/ECL_v${TRIAL}/"

PRETRAIN_CKPTS="/projects/ml4science/time_series/pretrain_checkpoints_upd/"
FINETUNE_CKPTS="/projects/ml4science/time_series/finetune_checkpoints_upd/"

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
        --encoder_depth 2 \
        --mask_ratio 0.50 \
        --encoder_num_heads 8 \
        --lr 0.001 \
        --batch_size 32 \
        --encoder_embed_dim 32 \
        --decoder_embed_dim 32 \
        --decoder_num_heads 16 \
        --encoder_num_heads 16 \
        --decoder_depth 2 \
        --project_name ecl_masking \
        --trial $TRIAL \
        --dropout 0.1 \
	    --pretrain_checkpoints_dir $PRETRAIN_CKPTS
    
    for pred_len in ${PRED_LEN_ARRAY[@]}; do
        python -u executor.py \
            --task_name finetune \
            --device $DEVICE \
            --root_path $root_path \
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
            --encoder_num_heads 16 \
            --encoder_embed_dim 32 \
            --lr 0.0001 \
            --dropout 0.1\
            --fc_dropout 0.0 \
            --batch_size 32 \
            --accum_iter 1 \
            --project_name ecl_masking \
            --output_path $OUTPUT_PATH \
            --trial $TRIAL \
	        --finetune_checkpoints_dir $FINETUNE_CKPTS
    done
    
done
