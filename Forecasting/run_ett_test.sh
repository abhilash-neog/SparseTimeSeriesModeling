DATASET="ETT"
SOURCE_FILE="ETTh1"
DEVICE=0
PRETRAIN_EPOCHS=100
FINETUNE_EPOCHS_FROZEN=10
FINETUNE_EPOCHS_NOT_FROZEN=20


# PRETRAIN

python -u executor.py --task_name pretrain --device $DEVICE --run_name "vtest_pretrain_${SOURCE_FILE}_mask_40" --source_filename $SOURCE_FILE --dataset $DATASET --max_epochs $PRETRAIN_EPOCHS --mask_ratio 0.40 --encoder_embed_dim 8 --decoder_depth 1 --encoder_num_heads 8


# FINETUNE WITH FROZEN ENCODER

python -u executor.py --task_name finetune --device $DEVICE --run_name "vtest_finetune_${SOURCE_FILE}_mask_40_FRZN" --pretrain_run_name "vtest_pretrain_${SOURCE_FILE}_mask_40" --freeze_encoder "True" --max_epochs $FINETUNE_EPOCHS_FROZEN --dataset $DATASET

python -u executor.py --task_name finetune --device $DEVICE --run_name "vtest_finetune_${SOURCE_FILE}_mask_40_NOT_FRZN" --pretrain_run_name "vtest_finetune_${SOURCE_FILE}_mask_40_FRZN" --freeze_encoder "False" --max_epochs $FINETUNE_EPOCHS_NOT_FROZEN --dataset $DATASET --pretrain_checkpoints_dir "./finetune_checkpoints/"