#!/usr/bin/bash

export MASTER_ADDR=10.144.144.2
export MASTER_PORT=8000
export NCCL_SOCKET_IFNAME=tun0
export GLOO_SOCKET_IFNAME=tun0

CUDA_VISIBLE_DEVICES=0 python3 testdata.py \
    --rank 0 \
    --world 2 \
    --config_file tasks/medusa_llama/config/vicuna_7b_config.json \
    --dataset /mnt/8288e0df-3273-4363-b94b-f3eeaa458b29/zhaoxiudi/Jupiter-main/data/vicuna_blog_eval/question.jsonl
