#!/bin/bash
## extract model representations

export CUDA_VISIBLE_DEVICES=0;

for dataset in 'vgg16/imagenet' 'vgg16/places365' 'vgg16/cars' 'vgg16/vggface' 'vgg16/segnet' 'vgg16/random_init' 'vgg16/cifar10'
do
  trained_on=$dataset  # 'mnist' or 'vgg16/imagenet'
  extract_dataset='pets'  # 'mnist' 'custom3D'
  model_folder='1' #'all'
  python extract.py --trained_on $trained_on --dataset $extract_dataset --model_folder $model_folder
done
