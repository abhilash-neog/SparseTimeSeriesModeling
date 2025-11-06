ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4
BACKBONE_REVIN=${5}

GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"
root_path_name="/raid/abhilash/updated_synthetic_datasets/ETTm2/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_ettm2_imputed.csv"

OUTPUT_PATH="./outputs_upd/Spline/${MASKINGTYPE}/ETTm2_v${TRIAL}/"
CHECKPOINT="/raid/abhilash/LSTM_ckpts_upd/Spline/"
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
          --model_id "ETTm2_${seq_len}_${pred_len}" \
          --model LSTM \
          --data ETTm2 \
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
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH \
          --dropout 0.005 \
          --misstsm 0 \
          --hidden_size 32 \
          --backbone_revin $BACKBONE_REVIN
    done
done