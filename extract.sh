#!/bin/bash
## extract model representations

export CUDA_VISIBLE_DEVICES=0;

dataset='mnist2class'
model_folder='all'

python extract.py --dataset $dataset --model_folder $model_folder
