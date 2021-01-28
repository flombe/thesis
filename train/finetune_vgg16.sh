#!/bin/bash

dataset='pets'  # fine-tune dataset
epochs=100
bs=58  # 12 # 22
lr=0.0001  # set in optimizer
run_name='ft_cifar10_pets'
pre_dataset='cifar10'  # pre_trained models

export CUDA_VISIBLE_DEVICES=0;
python finetune_vgg16.py --dataset $dataset --epochs $epochs --bs $bs --lr $lr --run_name $run_name --pre_dataset $pre_dataset
