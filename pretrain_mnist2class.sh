#!/bin/bash
## Pre-train on MNIST2class dataset

# run train.py with following arguments parsed
dataset='mnist2class'
epochs=200
bs=64
lr=0.0001
run_name='pre_train_adam'

python train.py --dataset $dataset --epochs $epochs --bs $bs --lr $lr --run_name $run_name
