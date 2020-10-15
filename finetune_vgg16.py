from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
import os
from os.path import join
import json
import pandas as pd

import train_utils
from vgg_arch import vgg16
import datasets

# like ID paper finetuning

# VGG-16-R
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
dataset_dir = join(root_dir, 'data', dataset_name)  # target data for ft
source_dir = join(root_dir, 'models', pretrain_dataset)
output_dir = join(source_dir, 'ft_' + dataset_name)  # new folder for fine-tuned models

if dataset_name == 'custom3D':
    n_out_classes = 40
    dataset = datasets.Custom3D(dataset_dir=dataset_dir, device=device)
    train_loader = dataset.get_train_loader(batch_size=bs)
    test_loader = dataset.get_test_loader(batch_size=bs)
    class_names = dataset.class_names

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'imgs': transforms.Compose([
        transforms.Resize((224, 224), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# 1200 samples - bs=12 --> 1batch is 0.01 epoch

dataloaders ={'train': train_loader, 'test':test_loader}
#
# image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x])
#                   for x in ['train', 'test']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs, shuffle=True, num_workers=4)
#                for x in ['train', 'test']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)

                # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        # save model
        torch.save(model, join(output_dir, 'model_' + str(epoch) + '.pt'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # save training data
    tags = ['train_loss', 'test_loss', 'train_acc', 'test_acc']
    vals = [train_loss, test_loss, train_acc, test_acc]
    train_stats = dict(zip(tags, vals))

    #### add other df columns
    stats_file_path = join(model_dir, run_name + '_train_stats.json')
    with open(stats_file_path, 'w+') as f:
        json.dump(train_stats, f)

    # create dataframe
    df = pd.DataFrame(train_stats)


    # file = open(join(output_dir, "training_data"), 'wb')
    # pickle.dump(training_data, file)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# check if pre-trained model already saved
# model_path = join(source_dir, 'model_vgg16_pre_imagenet.pt')
# if os.path.exists(model_path):
#     model_pre = torch.load(model_path)
# else:
#     # load & save pretrained model/weights
#     model_pre = vgg16(pretrained=True)  # pre-trained on imageNet
#     torch.save(model_pre, model_path)
#     print(model_path, ' saved.')


# check if pre-trained model already saved
model_path = join(source_dir, 'model_vgg16_random_init.pt')
model_pre = vgg16(pretrained=False)
torch.save(model_pre, model_path)
print(model_path, ' saved.')




# define finetune model - freeze pre-trained params and newly initialize last layers
model_ft = model_pre
for param in model_ft.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
model_ft.features._modules['28'] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

num_ftrs = model_ft.classifier._modules['0'].in_features
model_ft.classifier._modules['0'] = nn.Linear(num_ftrs, 4096)

num_ftrs = model_ft.classifier._modules['3'].in_features
model_ft.classifier._modules['3'] = nn.Linear(num_ftrs, 4096)

num_ftrs = model_ft.classifier._modules['6'].in_features
model_ft.classifier._modules['6'] = nn.Linear(num_ftrs, n_out_classes)

model_ft = model_ft.to(device)  # on cuda

criterion = nn.CrossEntropyLoss()

# Only trainable parameters will be optimized
# optimizer_ft = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-3)

# Syntax to train layers differentially (with different learning rates)
optimizer_ft = torch.optim.SGD([
    {'params': model_ft.features._modules['28'].parameters(), 'lr': 1e-4},
    {'params': model_ft.classifier.parameters(), 'lr': 1e-3}
], lr=1e-2, momentum=0.9)

# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)



# model_best = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=15)

# save best model
# torch.save(model_best, join(output_dir, 'model_ft.pt'))




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
