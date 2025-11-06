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
BACKBONE_REVIN=${11}

model_name=LSTM
model_id_name=weather
data_name=weather

GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/weather/"
root_path_name="/raid/abhilash/synthetic_datasets/weather/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_weather.csv"

OUTPUT_PATH="./outputs_mask_fraction_${TASK}/${MASKINGTYPE}/weather_v${TRIAL}/"
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
          --gt_data_path weather.csv \
          --model_id $model_id_name_$seq_len'_'$pred_len \
          --model $model_name \
          --data $data_name \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 21 \
          --c_out 21 \
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
          --skip_connection $SKIP \
          --misstsm $MTSM \
          --q_dim 64 \
          --k_dim 64 \
          --v_dim 64 \
          --dropout 0.005 \
          --hidden_size 128 \
          --num_layers 2 \
          --backbone_revin $BACKBONE_REVIN
    done
done

        #   --q_dim 64 \
        #   --k_dim 64 \
        #   --v_dim 64 \
        #   --dropout 0.005 \
        #   --hidden_size 128 \
        #   --num_layers 2
        #   --heads 1
        # all 1s