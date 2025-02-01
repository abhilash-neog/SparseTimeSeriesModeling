DEVICES=$1
epochs=$2
CHECKPOINT="/raid/abhilash/misstsm_layers/time_series/PatchTST_supervised/checkpoints/"
OUTPUT_PATH="./outputs/ETTh2/"

seq_len=336
model_name=PatchTST

root_path_name=/raid/abhilash/forecasting_datasets/ETT/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

random_seed=2021
for pred_len in 96 192 336 720; do
    python -u run_longExp.py \
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