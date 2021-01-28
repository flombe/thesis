#!/bin/bash
export CUDA_VISIBLE_DEVICES=0;

dataset='custom3D'
epochs=100
bs=12
lr=0.01  # standard 0.0001 # start with 0.01 and use lr_scheduler
run_name='pre_custom3D'
seeds=10

python train.py --dataset $dataset --epochs $epochs --bs $bs --lr $lr --run_name $run_name --seeds $seeds
