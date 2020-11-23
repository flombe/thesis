#!/bin/bash
export CUDA_VISIBLE_DEVICES=1;

dataset='mnist'  # fine-tune dataset
epochs=100
bs=64
lr=0.0001
run_name='ft_fashionmnist_mnist'
pre_dataset='fashionmnist'  # pre_trained models
# assume a 10 seed pre_trainings

python finetune.py --dataset $dataset --epochs $epochs --bs $bs --lr $lr --run_name $run_name --pre_dataset $pre_dataset
