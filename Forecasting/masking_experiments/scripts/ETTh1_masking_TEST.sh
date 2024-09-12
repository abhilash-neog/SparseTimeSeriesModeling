DATASET="ETTh1"
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=100

BASE_PATH="/raid/abhilash/synthetic_datasets/ETTh1/"
ROOT_PATHS=$1
DEVICE=$2
TRIAL=$3
MASKINGTYPE=$4
# PRED_LEN_LIST=$5

SOURCE_FILE="v${TRIAL}_${MASKINGTYPE}_etth1.csv"

GT_SOURCE_FILE="ETTh1.csv"
GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"

OUTPUT_PATH="./outputs/${MASKINGTYPE}/ETTh1_v${TRIAL}/"

# IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

for id in $ROOT_PATHS; do
    
    root_path="${BASE_PATH}${id}"
    # # PRETRAIN
    # python -u executor.py \
    #     --task_name pretrain \
    #     --device $DEVICE \
    #     --root_path $root_path \
    #     --run_name "v${TRIAL}_${MASKINGTYPE}_pretrain_${DATASET}_${id}" \
    #     --source_filename $SOURCE_FILE \
    #     --dataset $DATASET \
    #     --max_epochs $PRETRAIN_EPOCHS \
    #     --mask_ratio 0.50 \
    #     --lr 0.001 \
    #     --batch_size 16 \
    #     --encoder_depth 1 \
    #     --encoder_num_heads 8 \
    #     --encoder_embed_dim 32 \
    #     --decoder_depth 1 \
    #     --decoder_num_heads 8 \
    #     --decoder_embed_dim 32 \
    #     --project_name ett_masking \
    #     --trial $TRIAL \
    #     --dropout 0.05

    # FINETUNE WITH NON-FROZEN ENCODER
    for pred_len in 96 192 336 720; do
        python -u executor.py \
            --task_name finetune \
            --device $DEVICE \
            --root_path $root_path\
            --gt_root_path $GT_ROOT_PATH \
            --gt_source_filename $GT_SOURCE_FILE \
            --run_name "ft_v${TRIAL}_${MASKINGTYPE}_finetune_${DATASET}_PRED_${pred_len}_${id}" \
            --pretrain_run_name "" \
            --freeze_encoder "False" \
            --max_epochs $FINETUNE_EPOCHS \
            --dataset $DATASET \
            --pred_len $pred_len \
            --source_filename $SOURCE_FILE \
            --pretrain_ckpt_name ckpt_best.pth \
            --encoder_depth 1 \
            --encoder_num_heads 8 \
            --encoder_embed_dim 32 \
            --lr 0.0001 \
            --dropout 0.05 \
            --fc_dropout 0.0 \
            --batch_size 16 \
            --project_name ett_masking \
            --output_path $OUTPUT_PATH \
            --trial $TRIAL
    done
done