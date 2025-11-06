ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4
TASK=$5
MTSM_NORM=$6
EMBED=$7
LAYERNORM=$8
SKIP=$9
MTSM=${10}
MTSM_HEADS=${11}
BACKBONE_REVIN=${12}

model_name=LSTM
model_id_name=ETTm2
data_name=ETTm2

GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"
root_path_name="/raid/abhilash/synthetic_datasets/ETTm2/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_ettm2.csv"

OUTPUT_PATH="./outputs_mask_fraction_${TASK}/${MASKINGTYPE}/ETTm2_v${TRIAL}/"
CHECKPOINT="/raid/abhilash/misstsm_layers/time_series/LSTM/${TASK}/Masked/checkpoints/"
seq_len=96

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 96 192 336 720; do
        python -u run_longExp.py \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path ETTm2.csv \
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
          --batch_size 128 \
          --trial $TRIAL \
          --checkpoints $CHECKPOINT \
          --learning_rate 0.005 \
          --output_path $OUTPUT_PATH \
          --layernorm $LAYERNORM\
          --mtsm_embed $EMBED\
          --mtsm_norm $MTSM_NORM \
          --backbone_revin $BACKBONE_REVIN \
          --skip_connection $SKIP \
          --dropout 0.005 \
          --misstsm $MTSM \
          --q_dim 16 \
          --k_dim 16 \
          --v_dim 16 \
          --hidden_size 32 \
          --misstsm_heads $MTSM_HEADS \
          --num_layers 2
    done
done

        #   --q_dim 64 \
        #   --k_dim 64 \
        #   --v_dim 64 \
        #   --hidden_size 64 \
        #   --misstsm_heads 1 \
        #   --num_layers 2


        #   --batch_size 128 \
        #   --learning_rate 0.005 \
        #   --dropout 0.005 \
        #   --q_dim 21 \
        #   --k_dim 21 \
        #   --v_dim 21 \
        #   --hidden_size 42 \
        #   --misstsm_heads 1 \
        #   --num_layers 2