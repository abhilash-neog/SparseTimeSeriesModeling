export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/traffic/"
root_path_name="/raid/abhilash/synthetic_datasets/traffic/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_traffic_imputed_SAITS.csv"

OUTPUT_PATH="./outputs/SAITS/${MASKINGTYPE}/traffic_v${TRIAL}/"
CHECKPOINT="/raid/abhilash/AutoFormer_ckpts/SAITS/"

seq_len=336

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 96 192 336 720; do
        python -u run.py \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path traffic.csv \
          --model_id "traffic_336_${pred_len}" \
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
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH
    done
done