DEVICE=1
EPOCHS=200

# PRETRAIN
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
    --pretrain_dataset Epilepsy \
    --target_dataset Epilepsy