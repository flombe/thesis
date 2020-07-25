#!/bin/bash
## extract model representations

export CUDA_VISIBLE_DEVICES=0;

trained_on='mnist'
extract_dataset='fashionmnist'  # 'mnist'
model_folder='all' #'1'

python extract.py --trained_on $trained_on --dataset $extract_dataset --model_folder $model_folder
