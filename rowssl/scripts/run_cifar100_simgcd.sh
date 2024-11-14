# CUDA_VISIBLE_DEVICES=1 python -m model.train_simgcd_moco \
#     --dataset_name 'cifar100_LT' \
#     --batch_size 128 \
#     --grad_from_block 11 \
#     --epochs 200 \
#     --num_workers 4 \
#     --use_ssb_splits \
#     --sup_weight 0.35 \
#     --weight_decay 5e-5 \
#     --transform 'imagenet' \
#     --lr 0.1 \
#     --eval_funcs 'v2' \
#     --warmup_teacher_temp 0.07 \
#     --teacher_temp 0.04 \
#     --warmup_teacher_temp_epochs 30 \
#     --memax_weight 4 \
#     --exp_name CIFAR100_simgcd+moco+new_temp_scaling_001-1 \
#     --split_train_val "False" \
#     --imb_ratio 100 \
#     --rev 'consis' \
#     --min_tau 0.01 \
#     --max_tau 1.0 \

CUDA_VISIBLE_DEVICES=2 python -m model.train_simgcd_moco \
    --dataset_name 'cifar100_LT' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 4 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 4 \
    --exp_name CIFAR100_SimGCD+MoCo \
    --split_train_val "False" \
    --imb_ratio 100 \
    --rev 'consis' \