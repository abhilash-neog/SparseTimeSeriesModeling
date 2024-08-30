#!/bin/bash
#SBATCH -J ecltesting
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --time=8:00:00 # 24 hours
#SBATCH --gres=gpu:1

module reset
module load Anaconda3/2020.11
source activate ptst

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/env/lib
DEVICES=$1
epochs=$2
CHECKPOINT="/projects/ml4science/time_series/PatchTST_supervised/checkpoints_temp/"
OUTPUT_PATH="/projects/ml4science/time_series/PatchTST_supervised/outputs/ETTh1_temp/"

seq_len=336
model_name=PatchTST

root_path_name=/projects/ml4science/time_series/ts_forecasting_datasets/ETT/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021
for pred_len in 96 192 336 720; do
    python -u run_longExp_temp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --gpu $DEVICES \
      --train_epochs $epochs\
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --checkpoints $CHECKPOINT \
      --output_path $OUTPUT_PATH
      # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done