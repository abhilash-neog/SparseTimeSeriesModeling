#!/bin/bash
#SBATCH -J ecltesting
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q #dgx_normal_q #a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --time=10:00:00 # 24 hours
#SBATCH --gres=gpu:1

module reset
module load Anaconda3/2020.11
source activate env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/env/lib

BASE_PATH="/projects/ml4science/time_series/ts_synthetic_datasets/updated_synthetic_datasets/weather/"
ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4
PRED_LEN_LIST=$5

DATA_PATH="v${TRIAL}_${MASKINGTYPE}_weather_imputed_SAITS.csv"

# OUTPUT_PATH="/projects/ml4science/time_series/SimMTM/SAITS/outputs/weather/${MASKINGTYPE}/"

GT_ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/weather/"

PRETRAIN_CHECKPOINTS_DIR="/projects/ml4science/time_series/SimMTM/SAITS/pretrain_checkpoints_upd/"
FINETUNE_CHECKPOINTS_DIR="/projects/ml4science/time_series/SimMTM/SAITS/finetune_checkpoints_upd/"

OUTPUT_PATH="/projects/ml4science/time_series/SimMTM/outputs_upd/SAITS/${MASKINGTYPE}/weather_v${TRIAL}/"

IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

for id in $ROOT_PATHS; do

    root_path="${BASE_PATH}${id}"
    # python -u run.py \
    #     --task_name pretrain \
    #     --root_path $root_path \
    #     --data_path $DATA_PATH \
    #     --model_id Weather \
    #     --model SimMTM \
    #     --data Weather \
    #     --features M \
    #     --seq_len 336 \
    #     --e_layers 2 \
    #     --positive_nums 2 \
    #     --mask_rate 0.5 \
    #     --enc_in 21 \
    #     --dec_in 21 \
    #     --c_out 21 \
    #     --n_heads 8 \
    #     --d_model 64 \
    #     --d_ff 64 \
    #     --learning_rate 0.001 \
    #     --batch_size 8 \
    #     --train_epochs 50 \
    #     --trial $TRIAL \
    #     --pretrain_checkpoints $PRETRAIN_CHECKPOINTS_DIR \
    #     --gpu $DEVICES
    
    for pred_len in ${PRED_LEN_ARRAY[@]}; do
        python -u run.py \
            --task_name finetune \
            --gt_root_path $GT_ROOT_PATH \
            --is_training 1 \
            --root_path $root_path \
            --data_path $DATA_PATH \
            --gt_data_path weather.csv \
            --model_id Weather \
            --model SimMTM \
            --data Weather \
            --features M \
            --seq_len 336 \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --n_heads 8 \
            --d_model 64 \
            --d_ff 64 \
            --batch_size 16 \
            --trial $TRIAL \
            --pretrain_checkpoints $PRETRAIN_CHECKPOINTS_DIR \
            --checkpoints $FINETUNE_CHECKPOINTS_DIR \
            --output_path $OUTPUT_PATH \
            --gpu $DEVICES
    done
done