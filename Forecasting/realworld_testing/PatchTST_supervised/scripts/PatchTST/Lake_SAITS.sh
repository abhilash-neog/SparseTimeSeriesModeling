OUTPUT_PATH="./outputs/Lake/"
DEVICES=$1

CHECKPOINT="./ckpts/"

# GT_ROOT_PATH="/raid/abhilash22/forecasting_datasets/ETT/"

seq_len=21
model_name=PatchTST

# root_path_name="/raid/abhilash22/synthetic_datasets/ETTh2/"
root_path_name="/home/sepidehfatemi/MissTSM_FCR/"

# data_path_name="v${TRIAL}_${MASKINGTYPE}_etth2_imputed.csv"
data_path_name="FCR_SAITS.csv"

model_id_name=Lake
data_name=Lake

random_seed=2021

# for id in $ROOT_PATHS; do
    # root_path="${root_path_name}${id}"
    # root_path = "${root_path_name}"

for pred_len in 7 14 21; do
    python -u run_longExp.py \
        --target 'daily_median_chla_interp_ugL' \
        --freq d\
        --label_len 7\
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name_$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --seq_len $seq_len \
        --pred_len 14 \
        --enc_in 7 \
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
        --train_epochs 1\
        --itr 1 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --checkpoints $CHECKPOINT \
        --output_path $OUTPUT_PATH
done