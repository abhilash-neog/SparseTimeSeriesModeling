DEVICE=$1
TRIAL=$2
IMPUTATION=$3
EPOCHS=100
OUTPUT_PATH="./outputs/${IMPUTATION}/"
# # PRETRAIN
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
#     --target_dataset EMG \
#     --output_path $OUTPUT_PATH
    
for p in p2 p4 p6 p8; do

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
        --pretrain_dataset EMG \
        --target_dataset EMG \
        --fraction $p \
        --trial $TRIAL \
        --imputation $IMPUTATION

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
        --pretrain_dataset EMG \
        --target_dataset EMG \
        --fraction $p \
        --trial $TRIAL \
        --imputation $IMPUTATION \
        --output_path $OUTPUT_PATH
done