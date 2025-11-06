#!/bin/bash
#SBATCH -J etth2testing
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q #dgx_normal_q #a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --time=9:00:00 # 24 hours
#SBATCH --gres=gpu:1

module reset
module load Anaconda3/2020.11
source activate ptst

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/env/lib

ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

# GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"
CHECKPOINT="/projects/ml4science/time_series/DLinear/SAITS/checkpoints/"
GT_ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/ETT/"

root_path_name="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/ETTm1/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_ettm1_imputed_SAITS.csv"

OUTPUT_PATH="/projects/ml4science/time_series/DLinear/outputs/SAITS/${MASKINGTYPE}/ETTm1_v${TRIAL}/"

seq_len=336

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 96 192 336 720; do
      python -u run_longExp.py \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path_name \
        --gt_root_path $GT_ROOT_PATH \
        --gt_data_path ETTm1.csv \
        --model_id "ETTm1_${seq_len}_${pred_len}" \
        --model DLinear \
        --data ETTm1 \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --des 'Exp' \
        --itr 1 \
        --batch_size 8 \
        --learning_rate 0.0001 \
        --gpu $DEVICES \
        --trial $TRIAL \
        --checkpoints $CHECKPOINT \
        --output_path $OUTPUT_PATH
    done
done