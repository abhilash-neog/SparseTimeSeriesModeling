
# LAKE_CONFIGS=("barc_observed.json" "cram_observed.json" "fcr_modeled.json" "fcr_observed.json" "liro_observed.json" "mendota_modeled.json" "prla_observed.json" "prpo_observed.json" "sugg_observed.json" "sunapee_modeled.json" "took_observed.json")

# LAKE_NAMES=("barc_observed" "cram_observed" "fcr_modeled" "fcr_observed" "liro_observed" "mendota_modeled" "prla_observed" "prpo_observed" "sugg_observed" "sunapee_modeled" "took_observed")

# DEVICES=(0 1 2 3 4 5 6 7) #ranges from 0 to 7 lambdapgml

DEVICE=7
PRETRAIN_LAKE="combined_modeled"
MAX_EPOCHS=40
# LAKE_CONFIG="fcr_modeled.json"
# LAKE_NAME="fcr_modeled"

# LAKE_CONFIG="mendota_modeled.json"
# LAKE_NAME="mendota_modeled"

# LAKE_CONFIG="sunapee_modeled.json"
# LAKE_NAME="sunapee_modeled"

# LAKE_CONFIG="config.json"
# LAKE_NAME="combined_modeled"

LAKE_CONFIG="barc_observed.json"
LAKE_NAME="barc_observed"

# LAKE_CONFIG="cram_observed.json"
# LAKE_NAME="cram_observed"

# LAKE_CONFIG="fcr_observed.json"
# LAKE_NAME="fcr_observed"

# LAKE_CONFIG="liro_observed.json"
# LAKE_NAME="liro_observed"

# LAKE_CONFIG="prla_observed.json"
# LAKE_NAME="prla_observed"

# LAKE_CONFIG="prpo_observed.json"
# LAKE_NAME="prpo_observed"

# LAKE_CONFIG="sugg_observed.json"
# LAKE_NAME="sugg_observed"

# LAKE_CONFIG="took_observed.json"
# LAKE_NAME="took_observed"


# Pre-train tasks with Different Mask Ratios
# python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_pretrain_${LAKE_NAME}_mask_25" --task_name pretrain --mask_ratio 0.25 --device $DEVICE --max_epochs $MAX_EPOCHS --encoder_embed_dim 8 --zero_shot False
# python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_pretrain_${LAKE_NAME}_mask_50"  --task_name pretrain --mask_ratio 0.5 --device $DEVICE --max_epochs $MAX_EPOCHS --encoder_embed_dim 8 --zero_shot False
# python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_pretrain_${LAKE_NAME}_mask_75"  --task_name pretrain --mask_ratio 0.75 --device $DEVICE --max_epochs $MAX_EPOCHS --encoder_embed_dim 8 --zero_shot False

# counter to keep track of the device
# counter=0

# for ((i=0; i<${#LAKE_CONFIGS[@]}; i++)); do
#     LAKE_CONFIG=${LAKE_CONFIGS[i]}
#     LAKE_NAME=${LAKE_NAMES[i]}
#     DEVICE=${DEVICES[counter]}
    
# Finetune Experiment with Frozen Encoder
# python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_finetune_${LAKE_NAME}_mask_25"  --pretrain_run_name "v2_pretrain_${PRETRAIN_LAKE}_mask_25"  --task_name finetune --lookback 14 --freeze_encoder "True" --ckpt_name ckpt_latest_lookback_14.pth --device $DEVICE --max_epochs $MAX_EPOCHS --feature_wise_rmse "False" --encoder_embed_dim 8 #--n2one_ft True
python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_finetune_${LAKE_NAME}_mask_25"  --pretrain_run_name "v2_pretrain_${PRETRAIN_LAKE}_mask_25"  --task_name finetune --lookback 21 --freeze_encoder "True" --ckpt_name ckpt_latest_lookback_21.pth --device $DEVICE --max_epochs $MAX_EPOCHS --feature_wise_rmse "False" --encoder_embed_dim 8 --n2one_ft "True"

# python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_finetune_${LAKE_NAME}_mask_50"  --pretrain_run_name "v2_pretrain_${PRETRAIN_LAKE}_mask_50"  --task_name finetune --lookback 14 --freeze_encoder "True" --ckpt_name ckpt_latest_lookback_14.pth --device $DEVICE --max_epochs $MAX_EPOCHS --feature_wise_rmse "False" --encoder_embed_dim 8 #--n2one_ft True
python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_finetune_${LAKE_NAME}_mask_50"  --pretrain_run_name "v2_pretrain_${PRETRAIN_LAKE}_mask_50"  --task_name finetune --lookback 21 --freeze_encoder "True" --ckpt_name ckpt_latest_lookback_21.pth --device $DEVICE --max_epochs $MAX_EPOCHS --feature_wise_rmse "False" --encoder_embed_dim 8 --n2one_ft "True"

# python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_finetune_${LAKE_NAME}_mask_75"  --pretrain_run_name "v2_pretrain_${PRETRAIN_LAKE}_mask_75"  --task_name finetune --lookback 14 --freeze_encoder "True" --ckpt_name ckpt_latest_lookback_14.pth --device $DEVICE --max_epochs $MAX_EPOCHS --feature_wise_rmse "False" --encoder_embed_dim 8 #--n2one_ft True
python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_finetune_${LAKE_NAME}_mask_75"  --pretrain_run_name "v2_pretrain_${PRETRAIN_LAKE}_mask_75"  --task_name finetune --lookback 21 --freeze_encoder "True" --ckpt_name ckpt_latest_lookback_21.pth --device $DEVICE --max_epochs $MAX_EPOCHS --feature_wise_rmse "False" --encoder_embed_dim 8 --n2one_ft "True"

# Finetune Experiment without Frozen Encoder
python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_finetune_${LAKE_NAME}_mask_25"  --pretrain_run_name "v2_pretrain_${PRETRAIN_LAKE}_mask_25"  --task_name finetune --lookback 21 --freeze_encoder "False" --ckpt_name ckpt_latest_lookback_14_EncNotFrozen.pth --device $DEVICE --max_epochs $MAX_EPOCHS --feature_wise_rmse "False" --encoder_embed_dim 8 --n2one_ft "True"
python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_finetune_${LAKE_NAME}_mask_50"  --pretrain_run_name "v2_pretrain_${PRETRAIN_LAKE}_mask_50"  --task_name finetune --lookback 21 --freeze_encoder "False" --ckpt_name ckpt_latest_lookback_14_EncNotFrozen.pth --device $DEVICE --max_epochs $MAX_EPOCHS --feature_wise_rmse "False" --encoder_embed_dim 8 --n2one_ft "True"
python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_finetune_${LAKE_NAME}_mask_75"  --pretrain_run_name "v2_pretrain_${PRETRAIN_LAKE}_mask_75"  --task_name finetune --lookback 21 --freeze_encoder "False" --ckpt_name ckpt_latest_lookback_14_EncNotFrozen.pth --device $DEVICE --max_epochs $MAX_EPOCHS --feature_wise_rmse "False" --encoder_embed_dim 8 --n2one_ft "True"

# Downstream task (finetune) from scratch 
# python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_finetune_${LAKE_NAME}_scratch" --task_name finetune --lookback 14 --freeze_encoder "False" --load_pretrain False --ckpt_name ckpt_latest_lookback_14.pth --device $DEVICE --max_epochs $MAX_EPOCHS --feature_wise_rmse "False" --encoder_embed_dim 8 #--n2one_ft True
python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_finetune_${LAKE_NAME}_scratch" --task_name finetune --lookback 21 --freeze_encoder "False" --load_pretrain False --ckpt_name ckpt_latest_lookback_21.pth --device $DEVICE --max_epochs $MAX_EPOCHS --feature_wise_rmse "False" --encoder_embed_dim 8 --n2one_ft "True"
    
#     # Increment counter and handle wrapping around DEVICE_VALUES
#     counter=$(( (counter + 1) % ${#DEVICES[@]} ))

# done