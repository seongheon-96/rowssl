CUDA_VISIBLE_DEVICES=0 python -m model.train_bacon_bal \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name 'cifar100' \
    --labeled_classes 50 \
    --num_workers 2 \
    --epochs 200\
    --exp_name cifar_bacon_bal\


CUDA_VISIBLE_DEVICES=0 python -m model.train_bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name 'cifar100_LT' \
    --labeled_classes 50 \
    --num_workers 2 \
    --epochs 200\
    --exp_name cifar_bacon_imb100\
    --imb_ratio  100\
    --rev 'consis' \