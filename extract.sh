#!/bin/bash
## extract model representations

export CUDA_VISIBLE_DEVICES=0;

trained_on='imagenet'
extract_dataset='custom3D'  # 'mnist' 'fashionmnist'
model_folder='1' #'all'

python extract.py --trained_on $trained_on --dataset $extract_dataset --model_folder $model_folder
