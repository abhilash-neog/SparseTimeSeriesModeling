export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DEVICES=$1
TRIAL=$2
# /projects/ml4science/realworld_data_MissTSM/FCR
GT_ROOT_PATH="/projects/ml4science/realworld_data_MissTSM/FCR"
gt_data_path_name="FCR_missing.csv"

root_path_name="/projects/ml4science/realworld_data_MissTSM/FCR"
data_path_name="FCR_SAITS.csv"


CHECKPOINT="./model_checkpoints/"
OUTPUT_PATH="./outputs/FCR_v${TRIAL}/"

data_name=Lake
seq_len=21

for pred_len in 7 14 21; do
    python -u run.py \
    --target 'daily_median_chla_interp_ugL' 'daily_median_watertemp_interp_degC'\
    --freq d \
    --label_len 7 \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --gt_root_path $GT_ROOT_PATH \
    --gt_data_path $gt_data_path_name \
    --model_id $data_name_$seq_len'_'$pred_len \
    --model Autoformer \
    --data $data_name \
    --features MD \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 15 \
    --dec_in 15 \
    --c_out 15 \
    --des 'Exp' \
    --d_model 32 \
    --d_ff 128 \
    --itr 1 \
    --batch_size 8 \
    --gpu $DEVICES \
    --checkpoints $CHECKPOINT \
    --trial $TRIAL \
    --train_epochs 100\
    --checkpoints $CHECKPOINT \
    --output_path $OUTPUT_PATH
done
