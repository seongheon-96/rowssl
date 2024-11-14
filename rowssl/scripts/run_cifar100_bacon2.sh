CUDA_VISIBLE_DEVICES=2 python -m model.train_bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name 'cifar100_LT' \
    --labeled_classes 50 \
    --num_workers 2 \
    --epochs 200\
    --exp_name cifar_bacon_imb10_rev\
    --imb_ratio 10 \
    --rev 'reverse' \

CUDA_VISIBLE_DEVICES=2 python -m model.train_bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name 'cifar100_LT' \
    --labeled_classes 50 \
    --num_workers 2 \
    --epochs 200\
    --exp_name cifar_bacon_imb100_rev\
    --imb_ratio 100 \
    --rev 'reverse' \