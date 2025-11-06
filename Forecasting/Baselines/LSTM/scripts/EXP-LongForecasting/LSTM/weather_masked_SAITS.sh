ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/weather/"
root_path_name="/raid/abhilash/updated_synthetic_datasets/weather/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_weather_imputed_SAITS.csv"

OUTPUT_PATH="./outputs_upd/SAITS/${MASKINGTYPE}/weather_v${TRIAL}/"
CHECKPOINT="/raid/abhilash/LSTM_ckpts_upd/SAITS/"
seq_len=96

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 96 192 336 720; do
        python -u run_longExp.py \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path weather.csv \
          --model_id "Weather_${seq_len}_${pred_len}" \
          --model LSTM \
          --data weather \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 21 \
          --c_out 21 \
          --des 'Exp' \
          --itr 1 \
          --batch_size 128 \
          --gpu $DEVICES \
          --trial $TRIAL \
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH \
          --dropout 0.005 \
          --misstsm 0 \
          --learning_rate 0.005 \
          --hidden_size 128
    done
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