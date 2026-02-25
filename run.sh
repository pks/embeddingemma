#!/bin/bash


DIR=runs/curated_adaptive-sampling_mlp-head_tempdecay_hardneg_layer-1_batch1024_mean_1B

mkdir -p "$DIR"
source .venv/bin/activate

PYTHONPATH= CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
	--model MaLA-LM/emma-500-llama3-8b-bi \
	--dataset fineopus \
    --lang-set curated \
	--adaptive-sampling \
	--mlp-head \
	--mlp-hidden 2048 \
	--total-tokens 1B \
	--temp-start 0.5 \
	--temperature 0.05 \
    --neg-k 16 \
    --neg-warmup 10 \
	--neg-bank-size 8192 \
	--max-batch-tokens 1024 \
	--max-length 512 \
	--layer -1 \
	--out-dim 768 \
	--pooling mean \
    --train-device cuda:0 \
    --val-device cuda:0 \
    --val-size 1000 \
    --val-batch-tokens 512 \
    --checkpoint-tokens 10M \
    --seed 42 \
    --lr 2e-4 \
    --shuffle-buffer 500 \
	--output-dir "$DIR"
