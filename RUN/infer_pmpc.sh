#!/bin/bash
# PMPC Inference Script
# 논문 Implementation Details에 따른 inference 설정
#
# Inference parameters (논문 Section IV-A):
# - Top-k: k=30
# - Top-p: p=0.9
# - Temperature: τ=0.7
# - Repetition penalty: 1.03
# - Checkpoint: epoch-3 (lowest val PPL)

cd "$(dirname "$0")/.."

# Best checkpoint (lowest val PPL: 14.976)
CHECKPOINT="./DATA/pmpc.pmpc/2026-01-20145646.1e-05.16.1gpu/epoch-3.bin"

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --config_name pmpc \
    --inputter_name pmpc \
    --load_checkpoint "$CHECKPOINT" \
    --infer_input_file ./_reformat/test.txt \
    --infer_batch_size 16 \
    --max_input_length 140 \
    --max_decoder_input_length 40 \
    --max_length 50 \
    --min_length 5 \
    --temperature 0.7 \
    --top_k 30 \
    --top_p 0.9 \
    --repetition_penalty 1.03 \
    --seed 42
