export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4
PRED_LEN_LIST=$5

GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/weather/"
root_path_name="/raid/abhilash/updated_synthetic_datasets/weather/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_weather_imputed_SAITS.csv"

OUTPUT_PATH="./outputs/SAITS/${MASKINGTYPE}/weather_v${TRIAL}/"
PRETRAIN_CHECKPOINTS_DIR="/raid/abhilash/SimMTM/SAITS/pretrain_checkpoints_upd/"
FINETUNE_CHECKPOINTS_DIR="/raid/abhilash/SimMTM/SAITS/finetune_checkpoints_upd/"

seq_len=336

IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

for id in $ROOT_PATHS; do

    root_path="${BASE_PATH}${id}"
    python -u run.py \
        --task_name pretrain \
        --root_path $root_path \
        --data_path $DATA_PATH \
        --model_id Weather \
        --model SimMTM \
        --data Weather \
        --features M \
        --seq_len 336 \
        --e_layers 2 \
        --positive_nums 2 \
        --mask_rate 0.5 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --n_heads 8 \
        --d_model 64 \
        --d_ff 64 \
        --learning_rate 0.001 \
        --batch_size 8 \
        --train_epochs 50 \
        --trial $TRIAL \
        --pretrain_checkpoints $PRETRAIN_CHECKPOINTS_DIR \
        --gpu $DEVICES
    
    for pred_len in ${PRED_LEN_ARRAY[@]}; do
        python -u run.py \
            --task_name finetune \
            --gt_root_path $GT_ROOT_PATH \
            --is_training 1 \
            --root_path $root_path \
            --data_path $DATA_PATH \
            --gt_data_path weather.csv \
            --model_id Weather \
            --model SimMTM \
            --data Weather \
            --features M \
            --seq_len 336 \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --n_heads 8 \
            --d_model 64 \
            --d_ff 64 \
            --batch_size 16 \
            --trial $TRIAL \
            --train_epochs 10\
            --pretrain_checkpoints $PRETRAIN_CHECKPOINTS_DIR \
            --checkpoints $FINETUNE_CHECKPOINTS_DIR \
            --output_path $OUTPUT_PATH \
            --gpu $DEVICES
    done
done