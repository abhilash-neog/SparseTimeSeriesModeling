DEVICES=$1
TRIAL=$2

GT_ROOT_PATH="/raid/sepideh/Project_MissTSM/FCR/"
root_path_name="/raid/sepideh/Project_MissTSM/FCR/"
data_path_name="FCR_SAITS.csv"
gt_data_path_name="FCR_missing.csv"

CHECKPOINT="./model_checkpoints/"
OUTPUT_PATH="./outputs/FCR_v${TRIAL}/"

data_name=Lake
seq_len=21

for pred_len in 7 14 21; do
  python -u run_longExp.py \
    --target 'daily_median_chla_interp_ugL'\
    --freq d \
    --label_len 7 \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --gt_root_path $GT_ROOT_PATH \
    --gt_data_path $gt_data_path_name \
    --model_id $data_name_$seq_len'_'$pred_len \
    --model DLinear \
    --data $data_name \
    --features MD \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 15 \
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.3\
    --des 'Exp' \
    --itr 1 \
    --gpu $DEVICES \
    --batch_size 8 \
    --trial $TRIAL \
    --train_epochs 100\
    --learning_rate 0.05 \
    --checkpoints $CHECKPOINT \
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