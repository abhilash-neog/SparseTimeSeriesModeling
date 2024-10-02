
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/env/lib

DATASET="ETTh2"
PRETRAIN_EPOCHS=1
FINETUNE_EPOCHS=1

BASE_PATH="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/ETTh2/"
ROOT_PATHS=$1
TRIAL=$2
MASKINGTYPE=$3
PRED_LEN_LIST=$4
DEVICE=0

SOURCE_FILE="v${TRIAL}_${MASKINGTYPE}_etth2"

GT_SOURCE_FILE="ETTh2"
GT_ROOT_PATH="/projects/ml4science/time_series/ts_forecasting_datasets/ETT/"

OUTPUT_PATH="./outputs_new/${MASKINGTYPE}/ETTh2_v${TRIAL}/"

PRETRAIN_CHECKPOINTS_DIR="./pretrain_checkpoints_new/"
FINETUNE_CHECKPOINTS_DIR="./finetune_checkpoints_new/"

IFS=',' read -r -a PRED_LEN_ARRAY <<< "$PRED_LEN_LIST"

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
        --encoder_depth 2 \
        --encoder_num_heads 8 \
        --encoder_embed_dim 8 \
        --project_name ett_masking \
        --pretrain_checkpoints_dir $PRETRAIN_CHECKPOINTS_DIR \
        --trial $TRIAL

    # FINETUNE WITH NON-FROZEN ENCODER
    for pred_len in ${PRED_LEN_ARRAY[@]}; do
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
            --encoder_depth 2 \
            --encoder_num_heads 8 \
            --encoder_embed_dim 8 \
            --lr 0.0001 \
            --dropout 0.4 \
            --fc_dropout 0.4 \
            --batch_size 16 \
            --project_name ett_masking \
            --finetune_checkpoints_dir $FINETUNE_CHECKPOINTS_DIR \
            --pretrain_checkpoints_dir $PRETRAIN_CHECKPOINTS_DIR \
            --output_path $OUTPUT_PATH \
            --trial $TRIAL
    done
done