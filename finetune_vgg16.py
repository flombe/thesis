from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from os.path import join
import json
import pandas as pd

import train_utils
from vgg_arch import vgg16
import datasets

# like ID paper finetuning

# We removed the last hidden layers (the last convolutional and all the dense layers) of
# a VGG-16 network pre-trained on ImageNet and substituted it with randomly initialized
# layers of the same size except for the last hidden layer, in order to match the correct number of
# categories (40) of the custom dataset.
# We then fine-tuned it on the ≃ 85% of the data. More specifically we used 30 images for each category
# as training set and we tested on the remaining 6 images for each category.
# For the fine-tuning, we used a SGD with momentum 0.9, and a learning rate of 10−4
# in the last convolutional layer and of 10−3 in the dense layers. The other layers were kept frozen.
# The generalization performance after 15 epochs was ≈ 88% accuracy on the test set.


# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)

# parse args from sh script
parser = train_utils.train_args_parser()
parser.add_argument('--pre_dataset')  # add parser arg for pre-trained selection
args = parser.parse_args()
dataset_name, bs, epochs, lr, run_name, seeds = train_utils.parse_train_args(args)
pretrain_dataset = args.pre_dataset
print(' >> Run {run_name} on dataset {dataset} on pre-trained {pre} models. <<'.format(
    run_name=run_name, dataset=dataset_name, pre=pretrain_dataset))

# set dir
root_dir = os.getcwd()
print(root_dir)
dataset_dir = join(root_dir, 'data', dataset_name)  # target data for ft
source_dir = join(root_dir, 'models', 'vgg16', pretrain_dataset)
output_dir = join(source_dir, 'ft_' + dataset_name)  ## + '_3conv'  # new folder for fine-tuned models

if dataset_name == 'custom3D':
    n_out_classes = 40
    dataset = datasets.Custom3D(dataset_dir=dataset_dir, device=device)
    train_loader = dataset.get_train_loader(batch_size=bs)
    test_loader = dataset.get_test_loader(batch_size=bs)
    class_names = dataset.class_names


# load model
model_path = join(source_dir, f'model_vgg19_pre_{pretrain_dataset}.pt')
model_pre = torch.load(model_path)

# define finetune model - freeze pre-trained params and newly initialize last layers
print(model_pre)
model_ft = model_pre
for param in model_ft.parameters():
    param.requires_grad = False

if pretrain_dataset in ['imagenet', 'cifar10']:
    ## additional layers newly init
    # model_ft.features._modules['24'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # model_ft.features._modules['26'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # Parameters of newly constructed modules have requires_grad=True by default
    # model_ft.features._modules['28'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model_ft.features._modules['34'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    model_ft.classifier._modules['0'] = nn.Linear(512*7*7, 4096)  # hard-plug nr of feat
    model_ft.classifier._modules['3'] = nn.Linear(4096, 4096)
    model_ft.classifier._modules['6'] = nn.Linear(4096, n_out_classes)
else:
    # model_ft.features._modules['conv5_1'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # model_ft.features._modules['conv5_2'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    model_ft.features._modules['conv5_3'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    num_ftrs = model_ft.classifier._modules['fc6'].in_features
    model_ft.classifier._modules['fc6'] = nn.Linear(num_ftrs, 4096)

    num_ftrs = model_ft.classifier._modules['fc7'].in_features
    model_ft.classifier._modules['fc7'] = nn.Linear(num_ftrs, 4096)

    num_ftrs = model_ft.classifier._modules['fc8'].in_features
    model_ft.classifier._modules['fc8'] = nn.Linear(num_ftrs, n_out_classes)


model_ft = model_ft.to(device)  # on cuda

criterion = nn.CrossEntropyLoss()

# Only trainable parameters will be optimized
# optimizer_ft = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-3)

# Syntax to train layers differentially (with different learning rates)
optimizer_ft = torch.optim.SGD([
    {'params': model_ft.features.parameters(), 'lr': 1e-4},
    {'params': model_ft.classifier.parameters(), 'lr': 1e-3}
], lr=1e-2, momentum=0.9)

# Decay LR by a factor of 0.1 every 20 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)


seed = '1'

train_acc, train_loss, test_acc, test_loss, df = train_utils.train(model=model_ft, train_loader=train_loader,
                                                                   test_loader=test_loader, optimizer=optimizer_ft,
                                                                   device=device, criterion=criterion,
                                                                   epochs=epochs, output_dir=output_dir,
                                                                   run_name=run_name, seed=seed, ft=True,
                                                                   scheduler=exp_lr_scheduler)
print('Done fine-tuning run ', model_path)


# add to dff and save
df.insert(3, 'ft_dataset', dataset_name)
param = {'train_samples': len(train_loader)*bs,
         'batch_size': bs,
         'lr': 'layer dependent and decaying'}
df.insert(4, 'ft_param', [param] * len(df))
df.to_pickle(join(output_dir, 'df_' + run_name))

print(df)
print('Done fine-tuning all models.')
