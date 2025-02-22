ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4
TASK=$5
MTSM_NORM=$6
EMBED=$7
PRED_LEN_LIST=$8
LAYERNORM=$9

model_name=PatchTST

model_id_name=ETTm2
data_name=ETTm2

GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"
root_path_name="/raid/abhilash/synthetic_datasets/ETTm2/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_ettm2.csv"

OUTPUT_PATH="./outputs_mask_fraction_${TASK}/${MASKINGTYPE}/ETTm2_v${TRIAL}/"
CHECKPOINT="/raid/abhilash/misstsm_layers/time_series/PatchTST_supervised/${TASK}/Masked/checkpoints/"
seq_len=336

random_seed=2021

IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in ${PRED_LEN_ARRAY[@]}; do
        python -u run_longExp.py \
          --random_seed $random_seed \
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
          --e_layers 3 \
          --n_heads 16 \
          --d_model 128 \
          --q_dim 16 \
          --k_dim 8 \
          --v_dim 8 \
          --layernorm $LAYERNORM\
          --mtsm_embed $EMBED\
          --mtsm_norm $MTSM_NORM \
          --d_ff 256 \
          --dropout 0.2\
          --fc_dropout 0.2\
          --head_dropout 0\
          --patch_len 16\
          --stride 8\
          --des 'Exp' \
          --train_epochs 100\
          --patience 20\
          --lradj 'TST'\
          --gpu $DEVICES \
          --pct_start 0.4 \
          --itr 1 \
          --batch_size 128 \
          --trial $TRIAL \
          --learning_rate 0.0001 \
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH
          # >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
    done
done