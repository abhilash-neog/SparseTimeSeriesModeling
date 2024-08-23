export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

OUTPUT_PATH="./outputs/ECL_v"

ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/electricity/"
root_path_name="/raid/abhilash/synthetic_datasets/electricity/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_electricity_imputed.csv"

seq_len=336

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 96 192 336 720; do
        python -u run.py \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path electricity.csv \
          --model_id "ECL_336_${pred_len}" \
          --model Autoformer \
          --data electricity \
          --features M \
          --seq_len $seq_len \
          --label_len 168 \
          --pred_len $pred_len \
          --e_layers 2 \
          --d_layers 1 \
          --factor 3 \
          --enc_in 321 \
          --dec_in 321 \
          --c_out 321 \
          --des 'Exp' \
          --itr 1 \
          --gpu $DEVICES \
          --train_epochs 1 \
          --trial $TRIAL \
          --output_path "${OUTPUT_PATH}${TRIAL}/"
    done
done