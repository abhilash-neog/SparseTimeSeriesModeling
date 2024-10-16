export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u run.py \
    --task_name pretrain \
    --root_path /projects/ml4science/time_series/ts_forecasting_datasets/ETT/ \
    --data_path ETTh1.csv \
    --model_id ETTh1 \
    --model SimMTM \
    --data ETTh1 \
    --features M \
    --seq_len 336 \
    --e_layers 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 16 \
    --d_model 32 \
    --d_ff 64 \
    --positive_nums 3 \
    --mask_rate 0.5 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --train_epochs 50 \


