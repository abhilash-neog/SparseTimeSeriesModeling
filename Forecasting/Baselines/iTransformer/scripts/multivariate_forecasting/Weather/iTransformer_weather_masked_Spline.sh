export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/weather/"
root_path_name="/raid/abhilash/updated_synthetic_datasets/weather/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_weather_imputed.csv"

OUTPUT_PATH="./outputs_upd/Spline/${MASKINGTYPE}/weather_v${TRIAL}/"
CHECKPOINT="/raid/abhilash/iTransformer_ckpts_upd/Spline/"
seq_len=336

model_name=iTransformer

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 96 192 336 720; do
        python -u run.py \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path weather.csv \
          --model_id "weather_${seq_len}_${pred_len}" \
          --model $model_name \
          --data weather \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --e_layers 3 \
          --enc_in 21 \
          --dec_in 21 \
          --c_out 21 \
          --des 'Exp' \
          --d_model 512\
          --d_ff 512 \
          --itr 1 \
          --gpu $DEVICES \
          --trial $TRIAL \
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH
    done
done

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_192 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_336 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_720 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1