export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OUTPUT_PATH="./outputs/ETTm2_v"

DEVICES=$1
TRIAL=$2

root_path_name="/raid/abhilash/forecasting_datasets/ETT/"
data_path_name="ETTm2.csv"

seq_len=336

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