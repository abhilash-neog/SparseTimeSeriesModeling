#!/bin/bash
#SBATCH -J etth2testing
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q #dgx_normal_q #a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
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

CHECKPOINT="/projects/ml4science/time_series/DLinear/Spline/checkpoints/"
GT_ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/weather/"

root_path_name="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/weather/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_weather_imputed.csv"

OUTPUT_PATH="/projects/ml4science/time_series/DLinear/outputs/Spline/${MASKINGTYPE}/weather_v${TRIAL}/"

seq_len=336

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 96 192 336 720; do
        python -u run_longExp.py \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path weather.csv \
          --model_id "Weather_${seq_len}_${pred_len}" \
          --model DLinear \
          --data weather \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 21 \
          --des 'Exp' \
          --itr 1 \
          --batch_size 16 \
          --gpu $DEVICES \
          --trial $TRIAL \
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH
    done
done

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_$seq_len'_'192 \
#   --model DLinear \
#   --data ETTh2 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 192 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LongForecasting/DLinear_ETTh2_$seq_len'_'192.log

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_$seq_len'_'336 \
#   --model DLinear \
#   --data ETTh2 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 336 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LongForecasting/DLinear_ETTh2_$seq_len'_'336.log

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_$seq_len'_'720 \
#   --model DLinear \
#   --data ETTh2 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 720 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LongForecasting/DLinear_ETTh2_$seq_len'_'720.log