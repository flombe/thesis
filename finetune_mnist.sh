#!/bin/bash
## Fine-tune on MNIST2class model on MNIST data

# run finetune.py with following arguments parsed
epochs=200
bs=64
lr=0.0001

run_name='ft_mnist2_mnist_0batch0_'
model='model_pre_train_2class_0batch0.pt'
python finetune.py --epochs $epochs --bs $bs --lr $lr --run_name $run_name --model $model

run_name='ft_mnist2_mnist_0batch10_'
model='model_pre_train_2class_0batch10.pt'
python finetune.py --epochs $epochs --bs $bs --lr $lr --run_name $run_name --model $model

run_name='ft_mnist2_mnist_0batch500_'
model='model_pre_train_2class_0batch500.pt'
python finetune.py --epochs $epochs --bs $bs --lr $lr --run_name $run_name --model $model

run_name='ft_mnist2_mnist_1_'
model='model_pre_train_2class_1.pt'
python finetune.py --epochs $epochs --bs $bs --lr $lr --run_name $run_name --model $model

run_name='ft_mnist2_mnist_10_'
model='model_pre_train_2class_10.pt'
python finetune.py --epochs $epochs --bs $bs --lr $lr --run_name $run_name --model $model

run_name='ft_mnist2_mnist_100_'
model='model_pre_train_2class_100.pt'
python finetune.py --epochs $epochs --bs $bs --lr $lr --run_name $run_name --model $model

