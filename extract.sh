#!/bin/bash
## extract model representations

export CUDA_VISIBLE_DEVICES=1;

trained_on='mnist'  # 'mnist' or 'vgg16/imagenet'
extract_dataset='fashionmnist'  # 'mnist' 'custom3D'
model_folder='all' #'all'

python extract.py --trained_on $trained_on --dataset $extract_dataset --model_folder $model_folder
