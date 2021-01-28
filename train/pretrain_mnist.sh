#!/bin/bash
export CUDA_VISIBLE_DEVICES=0;

dataset='mnist_split2'
epochs=100
bs=64
lr=0.0001
run_name='pre_mnist_split2'
seeds=10

python train.py --dataset $dataset --epochs $epochs --bs $bs --lr $lr --run_name $run_name --seeds $seeds
