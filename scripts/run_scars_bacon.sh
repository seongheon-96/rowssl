CUDA_VISIBLE_DEVICES=7 python -m model.train_bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name 'scars' \
    --labeled_classes 100 \
    --num_workers 4 \
    --epochs 200\
    --exp_name scars_bacon_consis_imb2\
    --imb_ratio 2 \
    --rev 'consis' \
    --batch_size 128\

CUDA_VISIBLE_DEVICES=7 python -m model.train_bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name 'scars' \
    --labeled_classes 100 \
    --num_workers 4 \
    --epochs 200\
    --exp_name scars_bacon_rev_imb2\
    --imb_ratio 2 \
    --batch_size 128\
    --rev 'reverse' \
