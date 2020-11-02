#!/bin/bash
## extract model representations

export CUDA_VISIBLE_DEVICES=1;

trained_on='vgg16/segnet'  # 'mnist' or 'vgg16/imagenet'
extract_dataset='custom3D'  # 'mnist' 'fashionmnist'
model_folder='1' #'all'

python extract.py --trained_on $trained_on --dataset $extract_dataset --model_folder $model_folder
