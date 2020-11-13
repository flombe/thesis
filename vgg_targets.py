from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import os
from os.path import join
from torchvision import datasets, models, transforms
from datasets import Custom3D, Malaria
import pandas as pd
import numpy as np

custom3D_compare_plot = False
get_malaria = True

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)


if custom3D_compare_plot:

    # parse args from sh script
    pretrain_dataset = 'imagenet'
    dataset_name = 'custom3D'
    bs = 1

    # set dir
    root_dir = os.getcwd()
    dataset_dir = join(root_dir, 'data', dataset_name)  # target data for ft
    source_dir = join(root_dir, 'models', pretrain_dataset, 'vgg16')
    output_dir = join(source_dir, 'ft_' + dataset_name)  # new folder for fine-tuned models

    n_out_classes = 40

    dataset = Custom3D(dataset_dir=dataset_dir, device=device)
    class_names = dataset.class_names
    train_loader = dataset.get_train_loader(batch_size=bs, shuffle=False)
    test_loader = dataset.get_test_loader(batch_size=bs)

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


if get_malaria:
    pretrain_dataset = 'imagenet'
    dataset_name = 'malaria'
    bs = 10

    # set dirs
    root_dir = os.getcwd()
    dataset_dir = join(root_dir, 'data', dataset_name)  # target data for ft
    source_dir = join(root_dir, 'models', pretrain_dataset, 'vgg16')
    output_dir = join(source_dir, 'ft_' + dataset_name)  # new folder for fine-tuned models

    ## download from 'ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/malaria_cell_classification_code.zip'
    # import requests
    #
    # print('Download Starting...')
    # url = 'ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/malaria_cell_classification_code.zip'
    # r = requests.get(url)
    # with open(join(dataset_dir, 'second_try', 'malaria_data.zip'), 'wb') as output_file:
    #     output_file.write(r.content)
    # print('Download Completed!!!')

    # import zipfile
    # with zipfile.ZipFile(join(dataset_dir, 'archive.zip'), "r") as zip_ref:
    #     zip_ref.extractall("target")

    ## split into 80% train and 20% test folders
    # import splitfolders  # or import split_folders
    # # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    # splitfolders.ratio(join(dataset_dir, 'cell_images'), output=dataset_dir, seed=1, ratio=(.8, .2), group_prefix=None)

    dataset = Malaria(dataset_dir=dataset_dir, device=device)
    class_names = dataset.class_names
    print(class_names)
    train_loader = dataset.get_train_loader(batch_size=bs, shuffle=True)
    test_loader = dataset.get_test_loader(batch_size=bs)

    # visualize some malaria images
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    images, labels = next(iter(train_loader))

    fig = plt.figure(figsize=(25, 15))
    for i in range(10):
        print(images, labels)
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[], title=class_names[labels[i]])
        imshow(images[i])
    plt.show()