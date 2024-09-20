export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"
root_path_name="/raid/abhilash/synthetic_datasets/ETTh2/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_etth2.csv"

OUTPUT_PATH="./outputs/${MASKINGTYPE}/ETTh2_v${TRIAL}/"
CHECKPOINT="/raid/abhilash/MissTSM_iTransformer_ckpts/ETTh2/"
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
          --gt_data_path ETTh2.csv \
          --model_id "ETTh2_${seq_len}_${pred_len}" \
          --model $model_name \
          --data ETTh2 \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --e_layers 2 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --d_model 32 \
          --query_dim 128 \
          --d_ff 128 \
          --itr 1 \
          --gpu $DEVICES \
          --trial $TRIAL \
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH
    done
done