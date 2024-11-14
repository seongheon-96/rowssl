CUDA_VISIBLE_DEVICES=4 python -m model.train_bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name imagenet_100 \
    --labeled_classes 50 \
    --num_workers 4 \
    --epochs 100\
    --exp_name imagenet100_bacon_consis\
    --imb_ratio 100 \
    --rev 'consis' \

CUDA_VISIBLE_DEVICES=4 python -m model.train_bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name imagenet_100 \
    --labeled_classes 50 \
    --num_workers 4 \
    --epochs 100\
    --exp_name imagenet100_bacon_rev\
    --imb_ratio 100 \
    --rev 'reverse' \