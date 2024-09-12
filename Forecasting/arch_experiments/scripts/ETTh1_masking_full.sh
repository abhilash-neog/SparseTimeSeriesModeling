DATASET="ETTh1"
SOURCE_FILE="ETTh1.csv"

PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10

BASE_PATH="/raid/abhilash/synthetic_datasets/ETTh1/"
ROOT_PATHS=$1
DEVICE=$2
TRIAL=$3
MASKINGTYPE=$4

ENCODER_DIM=$5
ENCODER_DEPTH=$6
ENCODER_HEADS=$7

DECODER_DIM=$8
DECODER_DEPTH=$9
DECODER_HEADS=${10}

DROPOUT=${11}
FC_DROPOUT=${12}

kdim=${13}
vdim=${14}

SOURCE_FILE="v${TRIAL}_${MASKINGTYPE}_etth1.csv"
GT_SOURCE_FILE="ETTh1.csv"
GT_ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"

OUTPUT_PATH="./outputs_OG/${MASKINGTYPE}/ETTh1_v${TRIAL}_${ENCODER_DIM}_${ENCODER_DEPTH}_${ENCODER_HEADS}_${DECODER_DIM}_${DECODER_DEPTH}_${DECODER_HEADS}_dropout_${DROPOUT}_${FC_DROPOUT}_kdim_${kdim}_vdim_${vdim}/"
PRETRAIN_CKPT_DIR="/raid/abhilash/RUNS_OG_arch/pretrain_checkpoints_ETTh1/ckpt_${ENCODER_DIM}_${ENCODER_DEPTH}_${ENCODER_HEADS}_${DECODER_DIM}_${DECODER_DEPTH}_${DECODER_HEADS}_dropout_${DROPOUT}_${FC_DROPOUT}_kdim_${kdim}_vdim_${vdim}/"
FINETUNE_CKPT_DIR="/raid/abhilash/RUNS_OG_arch/finetune_checkpoints_ETTh1/ckpt_${ENCODER_DIM}_${ENCODER_DEPTH}_${ENCODER_HEADS}_${DECODER_DIM}_${DECODER_DEPTH}_${DECODER_HEADS}_dropout_${DROPOUT}_${FC_DROPOUT}_kdim_${kdim}_vdim_${vdim}/"

ROOT_PATH="/raid/abhilash/forecasting_datasets/ETT/"

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
        --encoder_depth $ENCODER_DEPTH \
        --decoder_depth $DECODER_DEPTH \
        --encoder_num_heads $ENCODER_HEADS \
        --encoder_embed_dim $ENCODER_DIM \
        --decoder_num_heads $DECODER_HEADS \
        --decoder_embed_dim $DECODER_DIM \
        --project_name ett \
        --pretrain_checkpoints_dir $PRETRAIN_CKPT_DIR \
        --dropout $DROPOUT \

    # FINETUNE WITH NON-FROZEN ENCODER
    for pred_len in 96 192 336 720; do
        python -u executor.py \
            --task_name finetune \
            --device $DEVICE \
            --root_path $root_path \
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
            --encoder_depth $ENCODER_DEPTH \
            --encoder_num_heads $ENCODER_HEADS \
            --encoder_embed_dim $ENCODER_DIM \
            --kdim $kdim \
            --vdim $vdim \
            --lr 0.0001 \
            --fc_dropout $FC_DROPOUT \
            --batch_size 16 \
            --project_name ett \
            --output_path $OUTPUT_PATH \
            --pretrain_checkpoints_dir $PRETRAIN_CKPT_DIR \
            --finetune_checkpoints_dir $FINETUNE_CKPT_DIR
    done
done