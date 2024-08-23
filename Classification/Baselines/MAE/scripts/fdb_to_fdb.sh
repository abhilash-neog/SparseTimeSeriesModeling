DEVICE=$1
EPOCHS=1

# PRETRAIN
python -u executor.py \
    --task_name pretrain \
    --device $DEVICE \
    --train_epochs $EPOCHS \
    --mask_ratio 0.50 \
    --lr 0.001 \
    --batch_size 16 \
    --encoder_depth 3 \
    --encoder_num_heads 16 \
    --encoder_embed_dim 32 \
    --pretrain_dataset FD-B \
    --target_dataset FD-B \
    
# FINETUNE
python -u executor.py \
    --task_name finetune \
    --device $DEVICE \
    --train_epochs $EPOCHS \
    --mask_ratio 0.50 \
    --lr 0.001 \
    --batch_size 16 \
    --encoder_depth 3 \
    --encoder_num_heads 16 \
    --encoder_embed_dim 32 \
    --pretrain_dataset FD-B \
    --target_dataset FD-B
    
# for p in p2 p4 p6 p8; do
#     python -u executor.py \
#         --task_name pretrain \
#         --device $DEVICE \
#         --train_epochs $EPOCHS \
#         --mask_ratio 0.50 \
#         --lr 0.001 \
#         --batch_size 16 \
#         --encoder_depth 3 \
#         --encoder_num_heads 16 \
#         --encoder_embed_dim 32 \
#         --pretrain_dataset FD-B \
#         --target_dataset FD-B \
#         --fraction $p

#     # FINETUNE
#     python -u executor.py \
#         --task_name finetune \
#         --device $DEVICE \
#         --train_epochs $EPOCHS \
#         --mask_ratio 0.50 \
#         --lr 0.001 \
#         --batch_size 16 \
#         --encoder_depth 3 \
#         --encoder_num_heads 16 \
#         --encoder_embed_dim 32 \
#         --pretrain_dataset FD-B \
#         --target_dataset FD-B \
#         --fraction $p
# done