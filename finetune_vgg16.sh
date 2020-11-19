#!/bin/bash
export CUDA_VISIBLE_DEVICES=1;  #####

dataset='pets'  # 'custom3D', 'malaria'  # fine-tune dataset
epochs=100
bs=58  # 12 # 22
lr=0.0001  # set in optimizer
run_name='ft_cifar10_pets'
pre_dataset='cifar10' # 'imagenet'  # pre_trained models

python finetune_vgg16.py --dataset $dataset --epochs $epochs --bs $bs --lr $lr --run_name $run_name --pre_dataset $pre_dataset
