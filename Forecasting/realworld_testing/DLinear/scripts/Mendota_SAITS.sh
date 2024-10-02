
DEVICES=$1
TRIAL=$2

GT_ROOT_PATH="/projects/ml4science/realworld_data_MissTSM/Mendota"
gt_data_path_name="Mendota_missing_daily.csv"

root_path_name="/projects/ml4science/realworld_data_MissTSM/Mendota"
data_path_name="Mendota_SAITS.csv"


CHECKPOINT="./model_checkpoints/"
OUTPUT_PATH="./outputs/Mendota_v${TRIAL}/"

data_name=Lake
seq_len=21

for pred_len in 7 14 21; do
  python -u run_longExp.py \
    --target 'avg_chlor_rfu' 'avg_do_wtemp'\
    --freq d \
    --label_len 7 \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --gt_root_path $GT_ROOT_PATH \
    --gt_data_path $gt_data_path_name \
    --model_id $data_name_$seq_len'_'$pred_len \
    --model DLinear \
    --data $data_name \
    --features MD \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 16 \
    --dec_in 16 \
    --c_out 16 \
    --d_ff 128 \
    --d_model 32 \
    --des 'Exp' \
    --itr 1 \
    --gpu $DEVICES \
    --batch_size 16 \
    --trial $TRIAL \
    --train_epochs 100\
    --learning_rate 0.05 \
    --checkpoints $CHECKPOINT \
    --output_path $OUTPUT_PATH
done