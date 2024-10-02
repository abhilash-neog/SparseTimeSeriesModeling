DATASET="FCR"
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=20

DEVICE=$1
enc_dim=$2
enc_num_heads=$3
enc_depth=$4

dec_dim=$5
dec_num_heads=$6
dec_depth=$7

OUTPUT_PATH="./outputs_${enc_dim}_${enc_num_heads}_${enc_depth}_${dec_dim}_${dec_num_heads}_${dec_depth}/FCR_v${TRIAL}/"
root_path_name="/raid/sepideh/Project_MissTSM/FCR/"
SOURCE_FILE="FCR_missing"

seq_len=21

# PRETRAIN
# python -u executor.py \
#     --task_name pretrain \
#     --device $DEVICE \
#     --target 'daily_median_chla_interp_ugL' 'daily_median_watertemp_interp_degC'\
#     --features MD\
#     --root_path $root_path_name \
#     --run_name "v${TRIAL}_pretrain_${DATASET}" \
#     --source_filename $SOURCE_FILE \
#     --dataset $DATASET \
#     --max_epochs $PRETRAIN_EPOCHS \
#     --mask_ratio 0.50 \
#     --lr 0.001 \
#     --enc_in 15 \
#     --batch_size 8 \
#     --seq_len $seq_len \
#     --encoder_depth $enc_depth \
#     --encoder_num_heads $enc_num_heads \
#     --encoder_embed_dim $enc_dim \
#     --decoder_depth $dec_depth \
#     --decoder_num_heads $dec_num_heads \
#     --decoder_embed_dim $dec_dim \
#     --project_name ett_masking \
#     --dropout 0.05

# FINETUNE WITH NON-FROZEN ENCODER
for pred_len in 7 14 21; do
    python -u executor.py \
        --task_name finetune \
        --device $DEVICE \
        --target 'daily_median_chla_interp_ugL' 'daily_median_watertemp_interp_degC'\
        --features MD\
        --root_path $root_path_name\
        --gt_root_path $root_path_name \
        --gt_source_filename $SOURCE_FILE \
        --run_name "v${TRIAL}_finetune_${enc_dim}_${enc_num_heads}_${enc_depth}_${DATASET}_PRED_${pred_len}" \
        --pretrain_run_name "v${TRIAL}_pretrain_${DATASET}" \
        --freeze_encoder "False" \
        --max_epochs $FINETUNE_EPOCHS \
        --dataset $DATASET \
        --seq_len $seq_len\
        --pred_len $pred_len \
        --source_filename $SOURCE_FILE \
        --pretrain_ckpt_name ckpt_best.pth \
        --encoder_depth $enc_depth \
        --encoder_num_heads $enc_num_heads \
        --encoder_embed_dim $enc_dim \
        --lr 0.0001 \
        --enc_in 15 \
        --dropout 0.05 \
        --fc_dropout 0.005 \
        --batch_size 8 \
        --project_name ett_masking \
        --output_path $OUTPUT_PATH
done