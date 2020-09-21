#!/bin/bash
## extract model representations

export CUDA_VISIBLE_DEVICES=0;

trained_on='mnist_noise'
extract_dataset='mnist_noise'  # 'mnist' 'fashionmnist'
model_folder='1' #'all'

python extract.py --trained_on $trained_on --dataset $extract_dataset --model_folder $model_folder
