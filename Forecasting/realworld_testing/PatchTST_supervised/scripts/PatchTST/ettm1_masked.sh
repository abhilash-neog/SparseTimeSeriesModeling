# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/LongForecasting" ]; then
#     mkdir ./logs/LongForecasting
# fi

BASE_PATH="/raid/abhilash22/forecasting_datasets/ETTm1/"

OUTPUT_PATH="./outputs/ETTm1/"
ROOT_PATHS=$1
DEVICES=$2
TRIAL=$3
MASKINGTYPE=$4

CHECKPOINT="./ckpts/"

GT_ROOT_PATH="/raid/abhilash22/forecasting_datasets/ETT/"

seq_len=336
model_name=PatchTST

root_path_name="/raid/abhilash22/synthetic_datasets/ETTm1/"
data_path_name="v${TRIAL}_${MASKINGTYPE}_ettm1_imputed.csv"
model_id_name=ETTm1
data_name=ETTm1

random_seed=2021

for id in $ROOT_PATHS; do
    root_path="${root_path_name}${id}"
    for pred_len in 96 192 336 720; do
        python -u run_longExp.py \
          --random_seed $random_seed \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path_name \
          --gt_root_path $GT_ROOT_PATH \
          --gt_data_path ETTm1.csv \
          --model_id $model_id_name_$seq_len'_'$pred_len \
          --model $model_name \
          --data $data_name \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --enc_in 7 \
          --e_layers 3 \
          --n_heads 16 \
          --d_model 128 \
          --d_ff 256 \
          --dropout 0.2\
          --fc_dropout 0.2\
          --head_dropout 0\
          --patch_len 16\
          --stride 8\
          --des 'Exp' \
          --train_epochs 1\
          --patience 20\
          --lradj 'TST'\
          --gpu $DEVICES \
          --pct_start 0.4\
          --itr 1 \
          --batch_size 128 \
          --learning_rate 0.0001 \
          --checkpoints $CHECKPOINT \
          --output_path $OUTPUT_PATH 
          #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
    done
done