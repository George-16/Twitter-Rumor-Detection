#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/test.py \
        --output_dir result \
        --pre_trained cardiffnlp/twitter-roberta-base-2021-124m \
        --ckpt result/f1_score=0.8727.ckpt \
        --dev_batch_size 16 \
        --test_batch_size 16 \
        --max_length 512 \
        --gpus 1 \
        --num_workers 16