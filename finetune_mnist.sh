#!/bin/bash
## Fine-tune on MNIST data

export CUDA_VISIBLE_DEVICES=0;

# run finetune.py with following arguments parsed
dataset='mnist'
epochs=100
bs=64
lr=0.0001
run_name='ft_mnist2_mnist_'
ft='mnist2class' # models to fine-tune
# assume a 10 seed pre_training

python finetune.py --dataset $dataset --epochs $epochs --bs $bs --lr $lr --run_name $run_name --ft $ft
