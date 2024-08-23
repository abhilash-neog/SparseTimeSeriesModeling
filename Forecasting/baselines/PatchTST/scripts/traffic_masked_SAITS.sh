#!/bin/bash
#SBATCH -J ecltesting
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q #dgx_normal_q #a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --time=21:00:00 # 24 hours
#SBATCH --gres=gpu:1
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

module reset
module load Anaconda3/2020.11
source activate ptst

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/abhilash22/.conda/envs/env/lib

ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

OUTPUT_PATH="/projects/ml4science/time_series/PatchTST_supervised/outputs/SAITS/${MASKINGTYPE}/traffic_v${TRIAL}/"

CHECKPOINT="/projects/ml4science/time_series/PatchTST_supervised/SAITS/checkpoints/"

GT_ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/traffic/"

seq_len=336
model_name=PatchTST

root_path_name="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/traffic/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_traffic_imputed_SAITS.csv"
model_id_name=traffic
data_name=traffic

random_seed=2021

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 720; do
        python -u run_longExp.py \
          --random_seed $random_seed \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path traffic.csv \
          --model_id $model_id_name_$seq_len'_'$pred_len \
          --model $model_name \
          --data $data_name \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 862 \
          --e_layers 3 \
          --n_heads 16 \
          --d_model 128 \
          --d_ff 256 \
          --dropout 0.2\
          --fc_dropout 0.2\
          --head_dropout 0\
          --patch_len 16\
          --stride 8\
          --des 'Exp' \
          --patience 10\
          --lradj 'TST'\
          --pct_start 0.2\
          --itr 1 \
          --batch_size 24 \
          --learning_rate 0.0001 \
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH \
          --gpu $DEVICES
          # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
    done
done