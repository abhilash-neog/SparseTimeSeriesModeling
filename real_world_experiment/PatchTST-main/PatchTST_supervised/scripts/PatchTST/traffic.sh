# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/LongForecasting" ]; then
#     mkdir ./logs/LongForecasting
# fi
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
BASE_PATH="/raid/abhilash22/forecasting_datasets/traffic/"

OUTPUT_PATH="./outputs/traffic/"
DEVICES=4
CHECKPOINT="./ckpts/"

seq_len=336
model_name=PatchTST

root_path_name=/raid/abhilash22/forecasting_datasets/traffic/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

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
      --train_epochs 1\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 \
      --batch_size 24 \
      --learning_rate 0.0001 \
      --checkpoints $CHECKPOINT \
      --output_path $OUTPUT_PATH
      # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done