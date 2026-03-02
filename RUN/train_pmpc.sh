#!/bin/bash
# PMPC Training Script - 논문 Implementation Details 완전 준수
#
# 논문 Section IV-A:
# - Backbone: BlenderBot-small
# - Prompt length: n = m = 20
# - Optimizer: AdamW (β1=0.9, β2=0.999, ε=1e-8)
# - Learning rate: 1e-5 with linear warmup (100 steps)
# - Batch size: 16
# - Max epochs: 8
# - Select checkpoint with lowest PPL on validation set

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config_name pmpc \
    --inputter_name pmpc \
    --eval_input_file ./_reformat/valid.txt \
    --seed 42 \
    --max_input_length 160 \
    --max_decoder_input_length 40 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --num_epochs 8 \
    --warmup_steps 100 \
    --fp16 false \
    --loss_scale 0.0 \
    --pbar true
