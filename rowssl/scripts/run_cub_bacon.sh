CUDA_VISIBLE_DEVICES=2 python -m model.train_bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name 'cub' \
    --labeled_classes 100 \
    --num_workers 4 \
    --epochs 200\
    --exp_name CUB_BACON_con\
    --imb_ratio 2 \
    --rev 'consis' \
    --batch_size 128\


