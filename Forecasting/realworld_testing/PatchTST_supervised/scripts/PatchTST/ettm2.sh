# if [ ! -d $LOG_PATH ]; then
#     mkdir $LOG_PATH
# fi

# if [ ! -d "${LOG_PATH}LongForecasting" ]; then
#     mkdir {$LOG_PATH}LongForecasting
# fi

BASE_PATH="/raid/abhilash22/forecasting_datasets/ETTm2/"

OUTPUT_PATH="./outputs/ETTm2/"
DEVICES=4
CHECKPOINT="./ckpts/"

seq_len=336
model_name=PatchTST

root_path_name=/raid/abhilash22/forecasting_datasets/ETT/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

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
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 1\
      --patience 20\
      --lradj 'TST'\
      --gpu $DEVICES \
      --pct_start 0.4 \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --checkpoints $CHECKPOINT \
      --output_path $OUTPUT_PATH
      # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done