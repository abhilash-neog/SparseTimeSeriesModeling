DATASET="Lake"
PRETRAIN_EPOCHS=1
FINETUNE_EPOCHS=1

DEVICE=$1
enc_dim=$2
enc_num_heads=$3
enc_depth=$4

OUTPUT_PATH="./outputs_${enc_dim}_${enc_num_heads}_${enc_depth}/Lake_v${TRIAL}/"
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
#     --encoder_depth 2 \
#     --encoder_num_heads 8 \
#     --encoder_embed_dim 8 \
#     --decoder_depth 2 \
#     --decoder_num_heads 8 \
#     --decoder_embed_dim 8 \
#     --project_name ett_masking \
#     --trial $TRIAL \
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
        --pretrain_run_name "" \
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