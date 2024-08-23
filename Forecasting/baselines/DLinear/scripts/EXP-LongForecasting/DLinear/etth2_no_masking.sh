# add --individual for DLinear-I
# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/LongForecasting" ]; then
#     mkdir ./logs/LongForecasting
# fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DEVICES=$1
TRIAL=$2

root_path_name="./datasets/ETT/"
data_path_name="ETTh2.csv"

OUTPUT_PATH="./outputs/ETTh2_v${TRIAL}/"

seq_len=336

for pred_len in 96 192 336 720; do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id "ETTh2_${seq_len}_${pred_len}" \
      --model DLinear \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 32 \
      --gpu $DEVICES \
      --trial $TRIAL \
      --learning_rate 0.05 \
      --output_path $OUTPUT_PATH
done

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_$seq_len'_'192 \
#   --model DLinear \
#   --data ETTh2 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 192 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LongForecasting/DLinear_ETTh2_$seq_len'_'192.log

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_$seq_len'_'336 \
#   --model DLinear \
#   --data ETTh2 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 336 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LongForecasting/DLinear_ETTh2_$seq_len'_'336.log

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_$seq_len'_'720 \
#   --model DLinear \
#   --data ETTh2 \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 720 \
#   --enc_in 7 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.05 >logs/LongForecasting/DLinear_ETTh2_$seq_len'_'720.log