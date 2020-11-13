#!/bin/bash
export CUDA_VISIBLE_DEVICES=0;

dataset='malaria'  # 'custom3D'  # fine-tune dataset
epochs=100
bs=22  # 12
lr=0.0001  # set in optimizer
run_name='ft_imagenet_malaria'
pre_dataset='imagenet'  # 'imagenet'  # pre_trained models

python finetune_vgg16.py --dataset $dataset --epochs $epochs --bs $bs --lr $lr --run_name $run_name --pre_dataset $pre_dataset
