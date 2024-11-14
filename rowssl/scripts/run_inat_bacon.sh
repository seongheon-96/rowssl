CUDA_VISIBLE_DEVICES=1 python -m model.bacon \
    --est_freq 10 \
    --ce_warmup 1 \
    --alpha 0 \
    --beta 0.5 \
    --dataset_name inaturelist18 \
    --labeled_classes 500 \
    --num_workers 4 \
    --epochs 70\
    --exp_name inat18-1k_Bacon_bacon_split_bacc_added\