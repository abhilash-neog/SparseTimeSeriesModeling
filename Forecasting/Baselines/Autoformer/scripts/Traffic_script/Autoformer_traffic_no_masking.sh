export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OUTPUT_PATH="./outputs/traffic_v"

DEVICES=$1
TRIAL=$2

root_path_name="/raid/abhilash/forecasting_datasets/traffic/"
data_path_name="traffic.csv"

seq_len=336

for pred_len in 96 192 336 720; do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id "Traffic_336_${pred_len}" \
      --model Autoformer \
      --data traffic \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --itr 1 \
      --gpu $DEVICES \
      --trial $TRIAL \
      --output_path "${OUTPUT_PATH}${TRIAL}/"
done