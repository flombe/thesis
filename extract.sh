#!/bin/bash
## extract model representations

export CUDA_VISIBLE_DEVICES=0;

trained_on='mnist_noise_struct'
extract_dataset='mnist_noise_struct'  # 'mnist' 'fashionmnist'
model_folder='all' #'1'

python extract.py --trained_on $trained_on --dataset $extract_dataset --model_folder $model_folder
