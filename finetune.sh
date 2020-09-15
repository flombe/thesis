#!/bin/bash
export CUDA_VISIBLE_DEVICES=0;

dataset='fashionmnist'  # fine-tune dataset
epochs=100
bs=64
lr=0.0001
run_name='ft_mnist_noise_struct_mnist'
pre_dataset='mnist_noise_struct'  # pre_trained models
# assume a 10 seed pre_training

python finetune.py --dataset $dataset --epochs $epochs --bs $bs --lr $lr --run_name $run_name --pre_dataset $pre_dataset
