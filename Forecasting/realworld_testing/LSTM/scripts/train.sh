PROJECT_NAME="project_masking_LSTM" 
DEVICE=$1

GT_ROOT_PATH="/raid/sepideh/Project_MissTSM/FCR/"
root_path_name="/raid/sepideh/Project_MissTSM/FCR/"


root_path_name="/projects/ml4science/realworld_data_MissTSM/FCR"
data_path_name="FCR_SAITS.csv"


data_name=Lake
seq_len=21


for pred_len in 7; do
    python -u executor.py \
    --project_name $PROJECT_NAME \
    --config_name "cram_observed.json" \
    --run_name "train_scratch_cram_obs_lstm" \
    --data_path $data_path_name \
    --task_name train \
    --horizon_csv_path './horizon_csv' \
    --lookback 21 \
    --horizon_window $pred_len \
    --device $DEVICE \
    --max_epochs 1 \
    --weight_decay 0.001 \
    --dropout 0.0005 \
    --ntrials 1
done