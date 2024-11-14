CUDA_VISIBLE_DEVICES=0 python -m model.train_simgcd \
    --dataset_name 'inaturelist18' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 70 \
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
    --exp_name inat18-1k_SimGCD_bacon_split_bacc_added \
    --split_train_val "False" \
