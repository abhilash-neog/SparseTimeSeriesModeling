DEVICE=7
PRETRAIN_LAKE="combined_modeled"
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
    
    
# Run zero-shot on lake
python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_zeroshot_${LAKE_NAME}_mask_25" --pretrain_run_name "v2_pretrain_${PRETRAIN_LAKE}_mask_25" --task_name zeroshot --lookback 21 --device $DEVICE --feature_wise_rmse "False" --n2one_ft "True"

# Run zero-shot on lake
python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_zeroshot_${LAKE_NAME}_mask_50" --pretrain_run_name "v2_pretrain_${PRETRAIN_LAKE}_mask_50" --task_name zeroshot --lookback 21 --device $DEVICE --feature_wise_rmse "False" --n2one_ft "True"

# Run zero-shot on lake
python -u executor.py --config_name $LAKE_CONFIG --run_name "v2_zeroshot_${LAKE_NAME}_mask_75" --pretrain_run_name "v2_pretrain_${PRETRAIN_LAKE}_mask_75" --task_name zeroshot --lookback 21 --device $DEVICE --feature_wise_rmse "False" --n2one_ft "True"