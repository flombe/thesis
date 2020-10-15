#!/bin/bash
export CUDA_VISIBLE_DEVICES=0;

dataset='custom3D'  # fine-tune dataset
epochs=100
bs=12
lr=0.0001
run_name='ft_random_init_custom3D'
pre_dataset='random_init'  # pre_trained models

python finetune_vgg16.py --dataset $dataset --epochs $epochs --bs $bs --lr $lr --run_name $run_name --pre_dataset $pre_dataset
