CUDA_VISIBLE_DEVICES=1 python -m model.train_ours \
    --dataset_name 'cub' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 4 \
    --use_ssb_splits False\
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 4 \
    --exp_name OLD_cub_ours_con2 \
    --imb_ratio 2\
    --rev 'consis'\
    --split_train_val "False" \