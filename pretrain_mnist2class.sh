#!/bin/bash
## Pre-train on MNIST2class dataset

export CUDA_VISIBLE_DEVICES=0; # restricts usage to device 1, the device will be mapped to id 0;
# to have it done automatically every time you log in, simply add this line to your .bashrc


# run train.py with following arguments parsed
dataset='mnist2class'
epochs=100
bs=64
lr=0.0001
run_name='pre_mnist2_'
seed=1

python train.py --dataset $dataset --epochs $epochs --bs $bs --lr $lr --run_name $run_name --seed $seed
