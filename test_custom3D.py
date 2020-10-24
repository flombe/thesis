from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import os
from os.path import join
from torchvision import datasets, models, transforms
from datasets import Custom3D
import pandas as pd

compare_plot = True

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)

# parse args from sh script
pretrain_dataset = 'imagenet'
dataset_name = 'custom3D'
bs = 1

# set dir
root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)  # target data for ft
source_dir = join(root_dir, 'models', pretrain_dataset, 'vgg16')
output_dir = join(source_dir, 'ft_' + dataset_name)  # new folder for fine-tuned models

if dataset_name == 'custom3D':
    n_out_classes = 40

dataset = Custom3D(dataset_dir=dataset_dir, device=device)
class_names = dataset.class_names
train_loader = dataset.get_train_loader(batch_size=bs, shuffle=False)
test_loader = dataset.get_test_loader(batch_size=bs)

if compare_plot:
    for i in range(1):
        image, label = next(iter(train_loader))
        print(label, class_names[label])
        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.title(class_names[label])
        plt.show()

    # transform: resize and to tensor
    data_transforms = transforms.Compose([transforms.Resize((224, 224), interpolation=2), transforms.ToTensor()])
    raw_train = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), data_transforms)
    for i in range(1):
        image, label = next(iter(raw_train))
        print(label)
        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.title(label)
        plt.show()

