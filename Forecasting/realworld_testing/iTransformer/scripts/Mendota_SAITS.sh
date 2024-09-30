export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DEVICES=$1
TRIAL=$2
# /projects/ml4science/realworld_data_MissTSM/Mendota
GT_ROOT_PATH="/projects/ml4science/realworld_data_MissTSM/Mendota"
gt_data_path_name="Mendota_missing_daily.csv"

root_path_name="/projects/ml4science/realworld_data_MissTSM/Mendota"
data_path_name="Mendota_SAITS.csv"


CHECKPOINT="./model_checkpoints/"
OUTPUT_PATH="./outputs/Mendota_v${TRIAL}/"

data_name=Lake
seq_len=21

model_name=iTransformer
cd ..

for pred_len in 7; do
    python -u run.py \
    --target 'avg_chlor_rfu' 'avg_do_wtemp'\
    --freq d \
    --label_len 7 \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --gt_root_path $GT_ROOT_PATH \
    --gt_data_path $gt_data_path_name \
    --model_id $data_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features MD \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --gpu $DEVICES \
    --checkpoints $CHECKPOINT \
    --trial $TRIAL \
    --train_epochs 5\
    --learning_rate 0.0001 \
    --checkpoints $CHECKPOINT \
    --output_path $OUTPUT_PATH
done


# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_192 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_336 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_720 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1