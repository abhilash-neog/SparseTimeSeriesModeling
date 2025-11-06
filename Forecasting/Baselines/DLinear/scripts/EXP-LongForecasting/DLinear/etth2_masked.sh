ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4
TASK=$5
MTSM_NORM=$6
EMBED=$7
PRED_LEN_LIST=$8
LAYERNORM=$9
SKIP=${10}
MTSM=${11}
MTSM_HEADS=${12}

model_name=DLinear
model_id_name=ETTh2
data_name=ETTh2

GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"
root_path_name="/raid/abhilash/synthetic_datasets/ETTh2/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_etth2.csv"

OUTPUT_PATH="./outputs_mask_fraction_${TASK}/${MASKINGTYPE}/ETTh2_v${TRIAL}/"
CHECKPOINT="/raid/abhilash/misstsm_layers/time_series/DLinear/${TASK}/Masked/checkpoints/"
seq_len=336

IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in ${PRED_LEN_ARRAY[@]}; do
        python -u run_longExp.py \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path ETTh2.csv \
          --model_id $model_id_name_$seq_len'_'$pred_len \
          --model $model_name \
          --data $data_name \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 7 \
          --des 'Exp' \
          --itr 1 \
          --gpu $DEVICES \
          --batch_size 32 \
          --trial $TRIAL \
          --checkpoints $CHECKPOINT \
          --learning_rate 0.05 \
          --output_path $OUTPUT_PATH \
          --layernorm $LAYERNORM\
          --mtsm_embed $EMBED\
          --mtsm_norm $MTSM_NORM \
          --skip_connection $SKIP \
          --misstsm $MTSM \
          --q_dim 16 \
          --k_dim 8 \
          --v_dim 8 \
          --misstsm_heads $MTSM_HEADS
    done
done