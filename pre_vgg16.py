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

dataset_name = 'custom3D'
pretrain_dataset = 'imagenet'

# set dir
root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)  # target data for ft
source_dir = join(root_dir, 'models', pretrain_dataset)
output_dir = join(source_dir, 'ft_' + dataset_name)  # new folder for fine-tuned models

if dataset_name == 'custom3D':
    n_out_classes = 40


image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x))
                  for x in ['train', 'test']}
# image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x])
#                   for x in ['train', 'test', 'imgs']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=12, shuffle=True, num_workers=4)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
print(class_names)



# check if pre-trained model already saved
model_path = join(source_dir, 'model_vgg16_pre_imagenet.pt')
if os.path.exists(model_path):
    model_pre = torch.load(model_path)
else:
    # load & save pretrained model/weights
    model_pre = vgg16(pretrained=True)  # pre-trained on imageNet
    torch.save(model_pre, model_path)
    print(model_path, ' saved.')
