
CUDA_VISIBLE_DEVICES=6 python -m model.train_gcd \
    --dataset_name 'scars' \
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
    --exp_name Scars_NIPS2024_gcd_rev \
    --split_train_val "False" \
    --imb_ratio 2 \
    --rev 'reverse' \

CUDA_VISIBLE_DEVICES=6 python -m model.train_gcd \
    --dataset_name 'scars' \
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
    --exp_name Scars_NIPS2024_gcd_con \
    --split_train_val "False" \
    --imb_ratio 2 \
    --rev 'consis' \


CUDA_VISIBLE_DEVICES=6 python -m model.train_gcd \
    --dataset_name 'scars' \
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
    --exp_name Scars_NIPS2024_gcd_rev \
    --split_train_val "False" \
    --imb_ratio 2 \
    --rev 'reverse' \

CUDA_VISIBLE_DEVICES=6 python -m model.train_gcd \
    --dataset_name 'scars' \
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
    --exp_name Scars_NIPS2024_gcd_con \
    --split_train_val "False" \
    --imb_ratio 2 \
    --rev 'consis' \