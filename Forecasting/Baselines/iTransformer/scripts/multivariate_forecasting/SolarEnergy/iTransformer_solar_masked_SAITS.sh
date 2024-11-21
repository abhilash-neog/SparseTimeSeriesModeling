export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/solar/"
root_path_name="/raid/abhilash/updated_synthetic_datasets/solar/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_solar_imputed_SAITS.csv"

OUTPUT_PATH="./outputs_upd/SAITS/${MASKINGTYPE}/Solar_v${TRIAL}/"
CHECKPOINT="/raid/abhilash/iTransformer_ckpts_upd/SAITS/"
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
          --gt_data_path solar.txt \
          --model_id "solar_${seq_len}_${pred_len}" \
          --model $model_name \
          --data Solar \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --e_layers 2 \
          --enc_in 137 \
          --dec_in 137 \
          --c_out 137 \
          --des 'Exp' \
          --d_model 512 \
          --d_ff 512 \
          --learning_rate 0.0005 \
          --itr 1 \
          --train_epochs 100\
          --gpu $DEVICES \
          --checkpoints $CHECKPOINT \
          --trial $TRIAL \
          --output_path $OUTPUT_PATH
    done
done
