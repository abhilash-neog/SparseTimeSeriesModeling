#!/bin/bash
#SBATCH -J mndta_iT
#SBATCH --account=ml4science
#SBATCH --mail-user=sepidehfatemi@vt.edu
#SBATCH --partition=dgx_normal_q #dgx_normal_q #a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --time=5:00:00 # 24 hours
#SBATCH --gres=gpu:1

module reset
module load Anaconda3/2020.11
source activate env

DEVICES=$1
TRIAL=$2

GT_ROOT_PATH="/projects/ml4science/realworld_data_MissTSM/Mendota"
gt_data_path_name="Mendota_missing_daily.csv"

root_path_name="/projects/ml4science/realworld_data_MissTSM/Mendota"
data_path_name="Mendota_SAITS.csv"


CHECKPOINT="./model_checkpoints/"
OUTPUT_PATH="./outputs/Mendota_v${TRIAL}/"


data_name=Lake
seq_len=21

random_seed=2023

for pred_len in 7 14 21; do
    python -u run_longExp.py \
        --target 'avg_chlor_rfu' 'avg_do_wtemp'\
        --freq d \
        --label_len 7 \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --gt_root_path $GT_ROOT_PATH \
        --gt_data_path $gt_data_path_name \
        --model_id $data_name_$seq_len'_'$pred_len \
        --model PatchTST \
        --data $data_name \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --features MD \
        --enc_in 16 \
        --dec_in 16 \
        --c_out 16 \
        --d_ff 128 \
        --d_model 32 \
        --e_layers 3 \
        --n_heads 4 \
        --dropout 0.3\
        --fc_dropout 0.3\
        --head_dropout 0\
        --patch_len 16\
        --stride 1\
        --des 'Exp' \
        --gpu $DEVICES \
        --train_epochs 100\
        --itr 1 \
        --batch_size 16 \
        --learning_rate 0.0001 \
        --checkpoints $CHECKPOINT \
        --output_path $OUTPUT_PATH \
        --trial $TRIAL
done