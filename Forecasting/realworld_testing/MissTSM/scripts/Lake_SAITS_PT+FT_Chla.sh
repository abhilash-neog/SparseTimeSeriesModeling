DATASET="Lake"
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

DEVICE=$1
TRIAL=$2

OUTPUT_PATH="./outputs_CL/Lake_v${TRIAL}/"
root_path_name="/raid/sepideh/Project_MissTSM/FCR/"
SOURCE_FILE="FCR_missing.csv"

IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"
seq_len=21

# PRETRAIN
python -u executor_test.py \
    --task_name pretrain \
    --device $DEVICE \
    --root_path $root_path_name \
    --run_name "v${TRIAL}_pretrain_CL_${DATASET}" \
    --source_filename $SOURCE_FILE \
    --dataset $DATASET \
    --max_epochs $PRETRAIN_EPOCHS \
    --mask_ratio 0.50 \
    --lr 0.001 \
    --enc_in 15 \
    --batch_size 8 \
    --seq_len $seq_len \
    --encoder_depth 1 \
    --encoder_num_heads 8 \
    --encoder_embed_dim 8 \
    --decoder_depth 1 \
    --decoder_num_heads 8 \
    --decoder_embed_dim 16 \
    --project_name ett_masking \
    --trial $TRIAL \
    --dropout 0.05

# FINETUNE WITH NON-FROZEN ENCODER
for pred_len in 7 14 21; do
    python -u executor_test.py \
        --task_name finetune \
        --device $DEVICE \
        --root_path $root_path_name\
        --gt_root_path $root_path_name \
        --gt_source_filename $SOURCE_FILE \
        --run_name "v${TRIAL}_finetune_${DATASET}_CL_PRED_${pred_len}" \
        --pretrain_run_name "v${TRIAL}_pretrain_CL_${DATASET}" \
        --freeze_encoder "False" \
        --max_epochs $FINETUNE_EPOCHS \
        --dataset $DATASET \
        --seq_len $seq_len\
        --pred_len $pred_len \
        --source_filename $SOURCE_FILE \
        --pretrain_ckpt_name ckpt_best.pth \
        --encoder_depth 1 \
        --encoder_num_heads 8 \
        --encoder_embed_dim 8 \
        --lr 0.0001 \
        --enc_in 15 \
        --dropout 0.05 \
        --fc_dropout 0.0 \
        --batch_size 8 \
        --project_name ett_masking \
        --output_path $OUTPUT_PATH \
        --trial $TRIAL
done