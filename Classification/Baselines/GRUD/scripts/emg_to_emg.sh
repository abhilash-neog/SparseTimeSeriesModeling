DEVICE=$1
TRIAL=$2
EPOCHS=1

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
#     --pretrain_dataset EMG \
#     --target_dataset EMG \
    
# # FINETUNE
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
#     --pretrain_dataset EMG \
#     --target_dataset EMG
    
for p in p2; do

    # FINETUNE
    python -u executor.py \
        --task_name finetune \
        --device $DEVICE \
        --finetune_epochs $EPOCHS \
        --mask_ratio 0.50 \
        --lr 0.001 \
        --batch_size 16 \
        --encoder_depth 3 \
        --encoder_num_heads 16 \
        --encoder_embed_dim 32 \
        --pretrain_dataset EMG \
        --target_dataset EMG \
        --fraction $p \
        --trial $TRIAL
done