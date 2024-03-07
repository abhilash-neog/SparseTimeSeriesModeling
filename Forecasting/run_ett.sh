DATASET="ETT"
SOURCE_FILE="ETTm2"
DEVICE=3
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=10


# PRETRAIN

python -u executor.py --task_name pretrain --device $DEVICE --run_name "v1_pretrain_${SOURCE_FILE}_mask_25" --source_filename $SOURCE_FILE --dataset $DATASET --max_epochs $PRETRAIN_EPOCHS --mask_ratio 0.25

python -u executor.py --task_name pretrain --device $DEVICE --run_name "v1_pretrain_${SOURCE_FILE}_mask_50" --source_filename $SOURCE_FILE --dataset $DATASET --max_epochs $PRETRAIN_EPOCHS --mask_ratio 0.50

python -u executor.py --task_name pretrain --device $DEVICE --run_name "v1_pretrain_${SOURCE_FILE}_mask_75" --source_filename $SOURCE_FILE --dataset $DATASET --max_epochs $PRETRAIN_EPOCHS --mask_ratio 0.75


# FINETUNE WITH FROZEN ENCODER

python -u executor.py --task_name finetune --device $DEVICE --run_name "v1_finetune_${SOURCE_FILE}_mask_25_FRZN" --pretrain_run_name "v1_pretrain_${SOURCE_FILE}_mask_25" --freeze_encoder "True" --max_epochs $FINETUNE_EPOCHS --dataset $DATASET

python -u executor.py --task_name finetune --device $DEVICE --run_name "v1_finetune_${SOURCE_FILE}_mask_50_FRZN" --pretrain_run_name "v1_pretrain_${SOURCE_FILE}_mask_50" --freeze_encoder "True" --max_epochs $FINETUNE_EPOCHS --dataset $DATASET

python -u executor.py --task_name finetune --device $DEVICE --run_name "v1_finetune_${SOURCE_FILE}_mask_75_FRZN" --pretrain_run_name "v1_pretrain_${SOURCE_FILE}_mask_75" --freeze_encoder "True" --max_epochs $FINETUNE_EPOCHS --dataset $DATASET


# FINETUNE WITHOUT FREEZING ENCODER

python -u executor.py --task_name finetune --device $DEVICE --run_name "v1_finetune_${SOURCE_FILE}_mask_25_NOT_FRZN" --pretrain_run_name "v1_pretrain_${SOURCE_FILE}_mask_25" --freeze_encoder "False" --max_epochs $FINETUNE_EPOCHS --dataset $DATASET

python -u executor.py --task_name finetune --device $DEVICE --run_name "v1_finetune_${SOURCE_FILE}_mask_50_NOT_FRZN" --pretrain_run_name "v1_pretrain_${SOURCE_FILE}_mask_50" --freeze_encoder "False" --max_epochs $FINETUNE_EPOCHS --dataset $DATASET

python -u executor.py --task_name finetune --device $DEVICE --run_name "v1_finetune_${SOURCE_FILE}_mask_75_NOT_FRZN" --pretrain_run_name "v1_pretrain_${SOURCE_FILE}_mask_75" --freeze_encoder "False" --max_epochs $FINETUNE_EPOCHS --dataset $DATASET