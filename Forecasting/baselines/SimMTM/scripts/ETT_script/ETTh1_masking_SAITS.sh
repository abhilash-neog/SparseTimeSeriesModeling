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

BASE_PATH="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/ETTh1/"
ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

DATA_PATH="v${TRIAL}_${MASKINGTYPE}_etth1_imputed_SAITS.csv"

OUTPUT_PATH="/projects/ml4science/time_series/SimMTM/SAITS/outputs/ETTh1/${MASKINGTYPE}/"

GT_ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/ETT/"

PRETRAIN_CHECKPOINTS_DIR="/projects/ml4science/time_series/SimMTM/SAITS/pretrain_checkpoints/"
FINETUNE_CHECKPOINTS_DIR="/projects/ml4science/time_series/SimMTM/SAITS/finetune_checkpoints/"

OUTPUT_PATH="/projects/ml4science/time_series/SimMTM/outputs/SAITS/${MASKINGTYPE}/ETTh1_v${TRIAL}/"

for id in $ROOT_PATHS; do

    root_path="${BASE_PATH}${id}"
    python -u run.py \
        --task_name pretrain \
        --root_path  $root_path \
        --data_path $DATA_PATH \
        --model_id ETTh1 \
        --model SimMTM \
        --data ETTh1 \
        --features M \
        --seq_len 336 \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 8 \
        --d_model 8 \
        --d_ff 32 \
        --positive_nums 3 \
        --mask_rate 0.5 \
        --learning_rate 0.001 \
        --batch_size 16 \
        --train_epochs 50 \
        --pretrain_checkpoints $PRETRAIN_CHECKPOINTS_DIR \
        --gpu $DEVICES
    
    for pred_len in 96 192 336 720; do
        python -u run.py \
            --task_name finetune \
            --gt_root_path $GT_ROOT_PATH \
            --root_path $root_path \
            --data_path $DATA_PATH \
            --gt_data_path ETTh1.csv \
            --is_training 1 \
            --model_id ETTh1 \
            --model SimMTM \
            --data ETTh1 \
            --features M \
            --seq_len 336 \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --n_heads 8 \
            --d_model 8 \
            --d_ff 32 \
            --dropout 0.4 \
            --head_dropout 0.2 \
            --batch_size 16 \
            --gpu $DEVICES \
            --pretrain_checkpoints $PRETRAIN_CHECKPOINTS_DIR \
            --checkpoints $FINETUNE_CHECKPOINTS_DIR \
            --output_path $OUTPUT_PATH \
            --train_epochs 10
    done
done