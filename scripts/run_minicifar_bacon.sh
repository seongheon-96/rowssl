CUDA_VISIBLE_DEVICES=1 python -m model.train_bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name 'cifar100_LT' \
    --labeled_classes 50 \
    --num_workers 4 \
    --epochs 200\
    --exp_name cifar_bacon_imb5 \
    --imb_ratio 5 \
    --rev 'consis' \

CUDA_VISIBLE_DEVICES=1 python -m model.train_bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name 'cifar100_LT' \
    --labeled_classes 50 \
    --num_workers 4 \
    --epochs 200\
    --exp_name cifar_bacon_imb10 \
    --imb_ratio 10 \
    --rev 'consis' \