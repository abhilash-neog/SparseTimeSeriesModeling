DEVICE=$1
TRIAL=$2
EPOCHS=100

# PRETRAIN
# python -u executor.py \
#     --task_name pretrain \
#     --device $DEVICE \
#     --train_epochs $EPOCHS \
#     --mask_ratio 0.50 \
#     --lr 0.001 \
#     --batch_size 16 \
#     --encoder_depth 3 \
#     --encoder_num_heads 16 \
#     --encoder_embed_dim 32 \
#     --pretrain_dataset Epilepsy \
#     --target_dataset Epilepsy
    
#     # FINETUNE
# python -u executor.py \
#     --task_name finetune \
#     --device $DEVICE \
#     --train_epochs $EPOCHS \
#     --mask_ratio 0.50 \
#     --lr 0.001 \
#     --batch_size 16 \
#     --encoder_depth 3 \
#     --encoder_num_heads 16 \
#     --encoder_embed_dim 32 \
#     --pretrain_dataset Epilepsy \
#     --target_dataset Epilepsy \
#     --output_path ./outputs/
   
for p in p8; do

    # PRETRAIN
    python -u executor.py \
        --task_name pretrain \
        --device $DEVICE \
        --pretrain_epochs $EPOCHS \
        --finetune_epochs $EPOCHS \
        --mask_ratio 0.50 \
        --lr 0.001 \
        --batch_size 64 \
        --encoder_depth 3 \
        --encoder_num_heads 16 \
        --encoder_embed_dim 32 \
        --pretrain_dataset SleepEEG \
        --target_dataset SleepEEG \
        --fraction $p \
        --trial $TRIAL
    
    # FINETUNE
    python -u executor.py \
        --task_name finetune \
        --device $DEVICE \
        --pretrain_epochs $EPOCHS \
        --finetune_epochs $EPOCHS \
        --mask_ratio 0.50 \
        --lr 0.001 \
        --batch_size 64 \
        --encoder_depth 3 \
        --encoder_num_heads 16 \
        --encoder_embed_dim 32 \
        --pretrain_dataset SleepEEG \
        --target_dataset SleepEEG \
        --output_path ./outputs/ \
        --fraction $p \
        --trial $TRIAL
done