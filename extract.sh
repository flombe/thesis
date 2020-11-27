#!/bin/bash
## extract model representations

export CUDA_VISIBLE_DEVICES=0;

#for dataset in 'vgg16/imagenet' 'vgg16/places365' 'vgg16/cars' 'vgg16/vggface' 'vgg16/segnet' 'vgg16/random_init' 'vgg16/cifar10'
for dataset in 'mnist' 'fashionmnist' 'mnist_split1' 'mnist_split2' 'mnist_noise_struct' 'mnist_noise'
do
  trained_on=$dataset  # 'mnist' or 'vgg16/imagenet'
  extract_dataset='fashionmnist'  # 'mnist' 'custom3D'
  model_folder='all' #'all'
  python extract.py --trained_on $trained_on --dataset $extract_dataset --model_folder $model_folder
done
