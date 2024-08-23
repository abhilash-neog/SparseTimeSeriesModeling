export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DEVICE=$1
TRIAL=$2
IMPUTATION=$3
EPOCHS=100
OUTPUT_PATH="./outputs/${IMPUTATION}/"
# python -u run.py \
#     --task_name pretrain \
#     --model SimMTM \
#     --model_id SimMTM \
#     --seq_len 336 \
#     --e_layers 3 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --n_heads 16 \
#     --d_model 32 \
#     --d_ff 64 \
#     --positive_nums 3 \
#     --mask_rate 0.5 \
#     --learning_rate 0.001 \
#     --batch_size 16 \
#     --train_epochs $EPOCHS \
#     --pretrain_dataset Epilepsy \
#     --target_dataset Epilepsy \
#     --gpu $DEVICE
    
# python -u run.py \
#     --task_name finetune \
#     --model SimMTM \
#     --model_id SimMTM \
#     --seq_len 336 \
#     --e_layers 3 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --n_heads 16 \
#     --d_model 32 \
#     --d_ff 64 \
#     --positive_nums 3 \
#     --mask_rate 0.5 \
#     --learning_rate 0.001 \
#     --batch_size 16 \
#     --train_epochs $EPOCHS \
#     --pretrain_dataset Epilepsy \
#     --target_dataset Epilepsy \
#     --gpu $DEVICE
    
for p in p8; do
    python -u run.py \
        --task_name pretrain \
        --model SimMTM \
        --model_id SimMTM \
        --seq_len 336 \
        --e_layers 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 64 \
        --positive_nums 3 \
        --mask_rate 0.5 \
        --learning_rate 0.001 \
        --batch_size 64 \
        --train_epochs $EPOCHS \
        --pretrain_dataset SleepEEG \
        --target_dataset SleepEEG \
        --fraction $p \
        --gpu $DEVICE \
        --trial $TRIAL \
        --imputation $IMPUTATION

    python -u run.py \
        --task_name finetune \
        --model SimMTM \
        --model_id SimMTM \
        --seq_len 336 \
        --e_layers 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --n_heads 16 \
        --d_model 32 \
        --d_ff 64 \
        --positive_nums 3 \
        --mask_rate 0.5 \
        --learning_rate 0.001 \
        --batch_size 64 \
        --train_epochs $EPOCHS \
        --pretrain_dataset SleepEEG \
        --target_dataset SleepEEG \
        --fraction $p \
        --gpu $DEVICE \
        --output_path $OUTPUT_PATH \
        --trial $TRIAL \
        --imputation $IMPUTATION
done