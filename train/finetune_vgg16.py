from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import os
from os.path import join

import train_utils
import datasets

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
dataset_dir = join(root_dir, '../data', dataset_name)  # target data for ft
source_dir = join(root_dir, '../models', 'vgg16', pretrain_dataset)
output_dir = join(source_dir, 'ft_' + dataset_name)

if dataset_name == 'custom3D':
    n_out_classes = 40
    dataset = datasets.Custom3D(dataset_dir=dataset_dir, device=device)
    train_loader = dataset.get_train_loader(batch_size=bs)
    test_loader = dataset.get_test_loader(batch_size=bs)
    class_names = dataset.class_names
if dataset_name == 'malaria':
    n_out_classes = 2
    dataset = datasets.Malaria(dataset_dir=dataset_dir, device=device)
    train_loader = dataset.get_train_loader(batch_size=bs)
    test_loader = dataset.get_test_loader(batch_size=bs)
    class_names = dataset.class_names
if dataset_name == 'pets':
    n_out_classes = 37
    dataset = datasets.Pets(dataset_dir=dataset_dir, device=device)
    train_loader = dataset.get_train_loader(batch_size=bs)
    test_loader = dataset.get_test_loader(batch_size=bs)
    class_names = dataset.class_names

# load model
if pretrain_dataset == 'segnet':    model_path = join(source_dir, f'model_vgg16bn_pre_{pretrain_dataset}.pt')
elif pretrain_dataset == 'cifar10': model_path = join(source_dir, f'model_vgg19_pre_{pretrain_dataset}.pt')
else:                               model_path = join(source_dir, f'model_vgg16_pre_{pretrain_dataset}.pt')
model_pre = torch.load(model_path)

# define finetune model - freeze pre-trained params and newly initialize last layers (standard grad=True)
print(model_pre)
model_ft = model_pre
for param in model_ft.parameters():
    param.requires_grad = False

if pretrain_dataset in ['imagenet', 'cifar10', 'segnet', 'random_init']:
    # Parameters of newly constructed modules have requires_grad=True by default
    if pretrain_dataset == 'cifar10':  # VGG-19
        model_ft.features._modules['34'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    elif pretrain_dataset == 'segnet':  # VGG-16bn
        model_ft.features._modules['40'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    else:
        # model_ft.features._modules['24'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # model_ft.features._modules['26'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model_ft.features._modules['28'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    model_ft.classifier._modules['0'] = nn.Linear(512*7*7, 4096)
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


# print how many layers are set to param.grad=True due to new initialization
j = 0
for param in model_ft.parameters():
    if param.requires_grad == True:
        j +=1
print(f'Layers with param=True: {j} / {len(list(model_ft.parameters()))}')

model_ft = model_ft.to(device)  # on cuda
criterion = nn.CrossEntropyLoss()

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
