CUDA_VISIBLE_DEVICES=4 python -m model.train_bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name 'herbarium_19' \
    --labeled_classes 341 \
    --num_workers 4 \
    --epochs 200\
    --exp_name herb_bacon2\
    --imb_ratio 1 \
    --rev 'consis' \

CUDA_VISIBLE_DEVICES=4 python -m model.train_bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name 'herbarium_19' \
    --labeled_classes 341 \
    --num_workers 4 \
    --epochs 200\
    --exp_name herb_bacon3\
    --imb_ratio 1 \
    --rev 'consis' \