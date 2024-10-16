#!/bin/bash
#SBATCH -J etth2testing
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q #dgx_normal_q #a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --time=5:00:00 # 24 hours
#SBATCH --gres=gpu:1

module reset
module load Anaconda3/2020.11
source activate env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/env/lib

OUTPUT_PATH="/projects/ml4science/time_series/Autoformer/outputs/ETTm2_v"

DEVICES=$1
TRIAL=$2

root_path_name="/projects/ml4science/time_series/ts_forecasting_datasets/ETT/"
data_path_name="ETTm2.csv"

seq_len=336

model_name=Autoformer

for pred_len in 96 192 336 720; do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id "ETTm2_336_${pred_len}" \
      --model Autoformer \
      --data ETTm2 \
      --features M \
      --seq_len $seq_len \
      --label_len 168 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 1 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --freq 't' \
      --itr 1 \
      --gpu $DEVICES \
      --output_path "${OUTPUT_PATH}${TRIAL}/"
      --trial $TRIAL
done