DEVICES=$1
TRIAL=$2

OUTPUT_PATH="./outputs/Lake_v${TRIAL}/"

CHECKPOINT="./ckpts/"

seq_len=21
model_name=PatchTST

GT_ROOT_PATH="/raid/sepideh/Project_MissTSM/FCR/"
root_path_name="/raid/sepideh/Project_MissTSM/FCR/"
data_path_name="FCR_SAITS.csv"
gt_data_path_name="FCR_missing.csv"

model_id_name=Lake
data_name=Lake

random_seed=2023
cd ../..
for pred_len in 7 14 21; do
    python -u run_longExp.py \
        --target 'daily_median_chla_interp_ugL' \
        --freq d \
        --label_len 7 \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --gt_root_path $GT_ROOT_PATH \
        --gt_data_path $gt_data_path_name \
        --model_id $model_id_name_$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 15 \
        --e_layers 3 \
        --n_heads 4 \
        --d_model 16 \
        --d_ff 128 \
        --dropout 0.3\
        --fc_dropout 0.3\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --gpu $DEVICES \
        --itr 1 \
        --batch_size 8 \
        --learning_rate 0.0001 \
        --checkpoints $CHECKPOINT \
        --output_path $OUTPUT_PATH \
        --trial $TRIAL
done