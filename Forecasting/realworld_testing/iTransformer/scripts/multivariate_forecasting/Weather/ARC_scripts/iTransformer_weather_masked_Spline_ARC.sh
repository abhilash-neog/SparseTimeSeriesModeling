#!/bin/bash
#SBATCH -J etth2testing
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q #dgx_normal_q #a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --time=3:00:00 # 24 hours
#SBATCH --gres=gpu:1

module reset
module load Anaconda3/2020.11
source activate env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/env/lib

ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

CHECKPOINT="/projects/ml4science/time_series/iTransformer/Spline/checkpoints/"
GT_ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/weather/"

root_path_name="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/weather/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_weather_imputed.csv"

OUTPUT_PATH="/projects/ml4science/time_series/iTransformer/outputs/Spline/${MASKINGTYPE}/weather_v${TRIAL}/"
seq_len=336

model_name=iTransformer

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 96 192 336 720; do
        python -u run.py \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path weather.csv \
          --model_id "weather_${seq_len}_${pred_len}" \
          --model $model_name \
          --data weather \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --e_layers 3 \
          --enc_in 21 \
          --dec_in 21 \
          --c_out 21 \
          --des 'Exp' \
          --d_model 512\
          --d_ff 512 \
          --itr 1 \
          --gpu $DEVICES \
          --trial $TRIAL \
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH
    done
done