#!/bin/bash
#SBATCH -J ecltesting
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q #dgx_normal_q #a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --time=20:00:00 # 24 hours
#SBATCH --gres=gpu:1

module reset
module load Anaconda3/2020.11
source activate env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/env/lib

BASE_PATH="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/electricity/"
ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4
PRED_LEN_LIST=$5

DATA_PATH="v${TRIAL}_${MASKINGTYPE}_electricity_imputed_SAITS.csv"

GT_ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/electricity/"

PRETRAIN_CHECKPOINTS_DIR="/projects/ml4science/time_series/SimMTM/SAITS/pretrain_checkpoints/"
FINETUNE_CHECKPOINTS_DIR="/projects/ml4science/time_series/SimMTM/SAITS/finetune_checkpoints/"

OUTPUT_PATH="/projects/ml4science/time_series/SimMTM/outputs/SAITS/${MASKINGTYPE}/ECL_v${TRIAL}/"

IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

for id in $ROOT_PATHS; do
    
    root_path="${BASE_PATH}${id}"
    python -u run.py \
        --task_name pretrain \
        --root_path $root_path \
        --data_path $DATA_PATH \
        --model_id ECL \
        --model SimMTM \
        --data ECL \
        --features M \
        --seq_len 336 \
        --label_len 48 \
        --e_layers 2 \
        --positive_nums 2 \
        --mask_rate 0.5 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --d_model 32 \
        --d_ff 64 \
        --n_heads 16 \
        --batch_size 32 \
        --train_epochs 1 \
        --temperature 0.02 \
        --trial $TRIAL \
        --pretrain_checkpoints $PRETRAIN_CHECKPOINTS_DIR \
        --gpu $DEVICES
    
    for pred_len in 96 192 336 720; do
        python -u run.py \
            --task_name finetune \
            --gt_root_path $GT_ROOT_PATH \
            --root_path $root_path \
            --data_path $DATA_PATH \
            --gt_data_path electricity.csv \
            --model_id ECL \
            --model SimMTM \
            --data ECL \
            --features M \
            --seq_len 336 \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --d_model 32 \
            --d_ff 64 \
            --n_heads 16 \
            --batch_size 32 \
            --gpu $DEVICES \
            --trial $TRIAL \
            --train_epochs 1\
            --pretrain_checkpoints $PRETRAIN_CHECKPOINTS_DIR \
            --checkpoints $FINETUNE_CHECKPOINTS_DIR \
            --output_path $OUTPUT_PATH
    done
done
