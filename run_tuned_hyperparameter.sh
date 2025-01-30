#!/bin/bash

set -x
data="ETTh1"
seq_len=512

# pred_len 96
echo "Running ETTh1 with prediction length: 96"
python run.py \
    --model tsmixer_rev_in \
    --data $data \
    --seq_len $seq_len \
    --pred_len 96 \
    --learning_rate 0.0001 \
    --n_block 6 \
    --dropout 0.9 \
    --ff_dim 512

# # pred_len 192
# echo "Running ETTh1 with prediction length: 192"
# python run.py \
#     --model tsmixer_rev_in \
#     --data $data \
#     --seq_len $seq_len \
#     --pred_len 192 \
#     --learning_rate 0.001 \
#     --n_block 4 \
#     --dropout 0.9 \
#     --ff_dim 256

# # pred_len 336
# echo "Running ETTh1 with prediction length: 336"
# python run.py \
#     --model tsmixer_rev_in \
#     --data $data \
#     --seq_len $seq_len \
#     --pred_len 336 \
#     --learning_rate 0.001 \
#     --n_block 4 \
#     --dropout 0.9 \
#     --ff_dim 256

# # pred_len 720
# echo "Running ETTh1 with prediction length: 720"
# python run.py \
#     --model tsmixer_rev_in \
#     --data $data \
#     --seq_len $seq_len \
#     --pred_len 720 \
#     --learning_rate 0.001 \
#     --n_block 2 \
#     --dropout 0.9 \
#     --ff_dim 64