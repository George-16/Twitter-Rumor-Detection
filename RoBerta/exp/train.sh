#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/train.py \
        --output_dir result \
        --pre_trained cardiffnlp/twitter-roberta-base-2021-124m \
        --warmup_ratio 0.1 \
        --learning_rate 5e-5 \
        --weight_decay 1e-3 \
        --gradient_clip_val 1 \
        --train_batch_size 16 \
        --dev_batch_size 16 \
        --test_batch_size 16 \
        --max_length 512 \
        --max_epochs 100 \
        --check_val_every_n_epoch 5 \
        --gpus 1 \
        --num_workers 16