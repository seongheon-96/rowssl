CUDA_VISIBLE_DEVICES=2 python -m model.train_bacon_cifar10 \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name 'cifar10' \
    --labeled_classes 5 \
    --num_workers 2 \
    --epochs 200\
    --exp_name CIFAR100_BACON_baconsplit\
    --imb_ratio  100\
    --batch_size 128\
    --rev consis \


