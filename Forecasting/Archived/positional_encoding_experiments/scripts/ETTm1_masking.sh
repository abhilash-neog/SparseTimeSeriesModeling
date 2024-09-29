DATASET="ETTm1"
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

BASE_PATH="/raid/abhilash/synthetic_datasets/ETTm1/"
ROOT_PATHS=$1
DEVICE=$2
TRIAL=$3
MASKINGTYPE=$4

SOURCE_FILE="v${TRIAL}_${MASKINGTYPE}_ettm1.csv"

GT_SOURCE_FILE="ETTm1.csv"
GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"

OUTPUT_PATH="./outputs/${MASKINGTYPE}/ETTm1_v${TRIAL}/"

# IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

for id in $ROOT_PATHS; do
    
    root_path="${BASE_PATH}${id}"
    # PRETRAIN
    python -u executor.py \
        --task_name pretrain \
        --device $DEVICE \
        --root_path $root_path \
        --run_name "v${TRIAL}_${MASKINGTYPE}_pretrain_${DATASET}_${id}" \
        --source_filename $SOURCE_FILE \
        --dataset $DATASET \
        --max_epochs $PRETRAIN_EPOCHS \
        --mask_ratio 0.50 \
        --lr 0.001 \
        --batch_size 16 \
        --encoder_depth 3 \
        --encoder_num_heads 8 \
        --encoder_embed_dim 32 \
        --decoder_depth 2 \
        --decoder_num_heads 8 \
        --decoder_embed_dim 32 \
        --project_name ett_masking \
        --trial $TRIAL \
        --dropout 0.1

    # FINETUNE WITH NON-FROZEN ENCODER
    for pred_len in 96 192 336 720; do
        python -u executor.py \
            --task_name finetune \
            --device $DEVICE \
            --root_path $root_path\
            --gt_root_path $GT_ROOT_PATH \
            --gt_source_filename $GT_SOURCE_FILE \
            --run_name "v${TRIAL}_${MASKINGTYPE}_finetune_${DATASET}_PRED_${pred_len}_${id}" \
            --pretrain_run_name "v${TRIAL}_${MASKINGTYPE}_pretrain_${DATASET}_${id}" \
            --freeze_encoder "False" \
            --max_epochs $FINETUNE_EPOCHS \
            --dataset $DATASET \
            --pred_len $pred_len \
            --source_filename $SOURCE_FILE \
            --pretrain_ckpt_name ckpt_best.pth \
            --encoder_depth 3 \
            --encoder_num_heads 8 \
            --encoder_embed_dim 32 \
            --lr 0.0001 \
            --dropout 0.1 \
            --fc_dropout 0.05 \
            --batch_size 32 \
            --project_name ett_masking \
            --output_path $OUTPUT_PATH \
            --trial $TRIAL
    done
done