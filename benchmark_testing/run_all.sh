
#########################################################################################################
##########################   FCR  ##########################
#########################################################################################################

# LAKE_CONFIG="fcr_modeled.json"
# LAKE_NAME="fcr_modeled"

# LAKE_CONFIG="mendota_modeled.json"
# LAKE_NAME="mendota_modeled"

# LAKE_CONFIG="sunapee_modeled.json"
# LAKE_NAME="sunapee_modeled"

LAKE_CONFIG="config.json"
LAKE_NAME="combined_modeled"

DEVICE=2
MAX_EPOCHS=40

# Pre-train tasks with Different Mask Ratios
python -u executor.py --config_name $LAKE_CONFIG --run_name "pretrain_${LAKE_NAME}_mask_25" --task_name pretrain --mask_ratio 0.25 --device $DEVICE --max_epochs $MAX_EPOCHS
python -u executor.py --config_name $LAKE_CONFIG --run_name "pretrain_${LAKE_NAME}_mask_50"  --task_name pretrain --mask_ratio 0.5 --device $DEVICE --max_epochs $MAX_EPOCHS
python -u executor.py --config_name $LAKE_CONFIG --run_name "pretrain_${LAKE_NAME}_mask_75"  --task_name pretrain --mask_ratio 0.75 --device $DEVICE --max_epochs $MAX_EPOCHS

# Finetune Experiment with Frozen Encoder
python -u executor.py --config_name $LAKE_CONFIG --run_name "finetune_${LAKE_NAME}_mask_25"  --pretrain_run_name "pretrain_${LAKE_NAME}_mask_25"  --task_name finetune --lookback 14 --freeze_encoder "True" --ckpt_name ckpt_latest_lookback_14.pth --device $DEVICE --max_epochs $MAX_EPOCHS
python -u executor.py --config_name $LAKE_CONFIG --run_name "finetune_${LAKE_NAME}_mask_25"  --pretrain_run_name "pretrain_${LAKE_NAME}_mask_25"  --task_name finetune --lookback 21 --freeze_encoder "True" --ckpt_name ckpt_latest_lookback_21.pth --device $DEVICE --max_epochs $MAX_EPOCHS

python -u executor.py --config_name $LAKE_CONFIG --run_name "finetune_${LAKE_NAME}_mask_50"  --pretrain_run_name "pretrain_${LAKE_NAME}_mask_50"  --task_name finetune --lookback 14 --freeze_encoder "True" --ckpt_name ckpt_latest_lookback_14.pth --device $DEVICE --max_epochs $MAX_EPOCHS
python -u executor.py --config_name $LAKE_CONFIG --run_name "finetune_${LAKE_NAME}_mask_50"  --pretrain_run_name "pretrain_${LAKE_NAME}_mask_50"  --task_name finetune --lookback 21 --freeze_encoder "True" --ckpt_name ckpt_latest_lookback_21.pth --device $DEVICE --max_epochs $MAX_EPOCHS

python -u executor.py --config_name $LAKE_CONFIG --run_name "finetune_${LAKE_NAME}_mask_75"  --pretrain_run_name "pretrain_${LAKE_NAME}_mask_75"  --task_name finetune --lookback 14 --freeze_encoder "True" --ckpt_name ckpt_latest_lookback_14.pth --device $DEVICE --max_epochs $MAX_EPOCHS
python -u executor.py --config_name $LAKE_CONFIG --run_name "finetune_${LAKE_NAME}_mask_75"  --pretrain_run_name "pretrain_${LAKE_NAME}_mask_75"  --task_name finetune --lookback 21 --freeze_encoder "True" --ckpt_name ckpt_latest_lookback_21.pth --device $DEVICE --max_epochs $MAX_EPOCHS

# Finetune Experiment without Frozen Encoder
python -u executor.py --config_name $LAKE_CONFIG --run_name "finetune_${LAKE_NAME}_mask_25"  --pretrain_run_name "pretrain_${LAKE_NAME}_mask_25"  --task_name finetune --lookback 14 --freeze_encoder "False" --ckpt_name ckpt_latest_lookback_14_EncNotFrozen.pth --device $DEVICE --max_epochs $MAX_EPOCHS
python -u executor.py --config_name $LAKE_CONFIG --run_name "finetune_${LAKE_NAME}_mask_50"  --pretrain_run_name "pretrain_${LAKE_NAME}_mask_50"  --task_name finetune --lookback 14 --freeze_encoder "False" --ckpt_name ckpt_latest_lookback_14_EncNotFrozen.pth --device $DEVICE --max_epochs $MAX_EPOCHS
python -u executor.py --config_name $LAKE_CONFIG --run_name "finetune_${LAKE_NAME}_mask_75"  --pretrain_run_name "pretrain_${LAKE_NAME}_mask_75"  --task_name finetune --lookback 14 --freeze_encoder "False" --ckpt_name ckpt_latest_lookback_14_EncNotFrozen.pth --device $DEVICE --max_epochs $MAX_EPOCHS

# Downstream task (finetune) from scratch 
python -u executor.py --config_name $LAKE_CONFIG --run_name "finetune_${LAKE_NAME}_scratch" --task_name finetune --lookback 14 --freeze_encoder "False" --load_pretrain False --ckpt_name ckpt_latest_lookback_14.pth --device $DEVICE --max_epochs $MAX_EPOCHS
python -u executor.py --config_name $LAKE_CONFIG --run_name "finetune_${LAKE_NAME}_scratch" --task_name finetune --lookback 21 --freeze_encoder "False" --load_pretrain False --ckpt_name ckpt_latest_lookback_21.pth --device $DEVICE --max_epochs $MAX_EPOCHS