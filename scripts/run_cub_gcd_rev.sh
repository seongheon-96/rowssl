CUDA_VISIBLE_DEVICES=4 python -m model.train_gcd \
    --dataset_name 'cub' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 4 \
    --use_ssb_splits  \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 4 \
    --exp_name CUB_GCD_rev \
    --split_train_val "False" \
    --imb_ratio 2 \
    --rev 'reverse' \


# CUDA_VISIBLE_DEVICES=3 python -m model.train_gcd \
#     --dataset_name 'cub' \
#     --batch_size 128 \
#     --grad_from_block 11 \
#     --epochs 200 \
#     --num_workers 4 \
#     --use_ssb_splits  \
#     --sup_weight 0.35 \
#     --weight_decay 5e-5 \
#     --transform 'imagenet' \
#     --lr 0.1 \
#     --eval_funcs 'v2' \
#     --warmup_teacher_temp 0.07 \
#     --teacher_temp 0.04 \
#     --warmup_teacher_temp_epochs 30 \
#     --memax_weight 4 \
#     --exp_name cub_GCD_OLDREAL_rev \
#     --split_train_val "False" \
#     --imb_ratio 2 \
#     --rev 'reverse' \


