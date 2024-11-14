CUDA_VISIBLE_DEVICES=3 python -m model.train_gcd_cifar10 \
    --dataset_name 'cifar10_LT' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
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
    --exp_name cifar10_GCD_rev_oursplit \
    --split_train_val "False" \
    --imb_ratio 100 \
    --rev 'reverse' \

CUDA_VISIBLE_DEVICES=3 python -m model.train_gcd_cifar10 \
    --dataset_name 'cifar10_LT' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
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
    --exp_name cifar10_GCD_consis \
    --split_train_val "False" \
    --imb_ratio 100 \
    --rev 'consis' \


CUDA_VISIBLE_DEVICES=3 python -m model.train_gcd_cifar10 \
    --dataset_name 'cifar10_LT' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
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
    --exp_name cifar10_GCD_rev_oursplit \
    --split_train_val "False" \
    --imb_ratio 100 \
    --rev 'reverse' \

CUDA_VISIBLE_DEVICES=3 python -m model.train_gcd_cifar10 \
    --dataset_name 'cifar10_LT' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
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
    --exp_name cifar10_GCD_consis \
    --split_train_val "False" \
    --imb_ratio 100 \
    --rev 'consis' \

