from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import os
from os.path import join
from torchvision import datasets, transforms
from datasets import Custom3D, Malaria, Pets
import numpy as np
from train.train_utils import set_seed

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)

# visualize images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# to-do: refactor in get and plot functions()
######
custom3D_compare_plot = True
get_malaria = False
get_pets = False
get_pets2 = False
print_pets = False
######


if custom3D_compare_plot:

    # parse args from sh script
    pretrain_dataset = 'imagenet'
    dataset_name = 'custom3D'
    bs = 6

    # set dir
    root_dir = os.getcwd()
    dataset_dir = join(root_dir, '../data', dataset_name)  # target data for ft
    source_dir = join(root_dir, '../models', pretrain_dataset, 'vgg16')
    output_dir = join(source_dir, 'ft_' + dataset_name)  # new folder for fine-tuned models

    n_out_classes = 40

    dataset = Custom3D(dataset_dir=dataset_dir, device=device)
    class_names = dataset.class_names
    train_loader = dataset.get_train_loader(batch_size=bs, shuffle=True)
    test_loader = dataset.get_test_loader(batch_size=bs)

    images, labels = next(iter(train_loader))

    fig = plt.figure(figsize=(9, 7))
    for i in range(6):
        ax = fig.add_subplot(2, 3, i + 1, xticks=[], yticks=[], title=class_names[labels[i]])
        imshow(images[i])
    plt.show()

    # transform: resize and to tensor
    data_transforms = transforms.Compose([transforms.Resize((224, 224), interpolation=2), transforms.ToTensor()])
    raw_train = datasets.ImageFolder(os.path.join(dataset_dir, '../train'), data_transforms)
    iter = iter(raw_train)
    fig = plt.figure(figsize=(9, 7))
    for i in range(6):
        image, label = next(iter)
        print(label)
        ax = fig.add_subplot(2, 3, i + 1, xticks=[], yticks=[], title=class_names[label])
        plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.tight_layout()
    plt.show()


if get_malaria:
    ## https://lhncbc.nlm.nih.gov/publication/pub9932

    pretrain_dataset = 'imagenet'
    dataset_name = 'malaria'
    bs = 6

    # set dirs
    root_dir = os.getcwd()
    dataset_dir = join(root_dir, '../data', dataset_name)  # target data for ft
    source_dir = join(root_dir, '../models', pretrain_dataset, 'vgg16')
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
    train_loader = dataset.get_train_loader(batch_size=bs, shuffle=True)
    test_loader = dataset.get_test_loader(batch_size=bs)

    images, labels = next(iter(train_loader))

    fig = plt.figure(figsize=(9, 7))
    for i in range(6):
        ax = fig.add_subplot(2, 3, i + 1, xticks=[], yticks=[], title=class_names[labels[i]])
        imshow(images[i])
    plt.show()


if get_pets:
    ## https://github.com/Skuldur/Oxford-IIIT-Pets-Pytorch/archive/master.zip
    ## https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz

    set_seed(1)

    pretrain_dataset = 'imagenet'
    dataset_name = 'pets'
    bs = 6

    # set dirs
    root_dir = os.getcwd()
    dataset_dir = join(root_dir, '../data', dataset_name)  # target data for ft
    source_dir = join(root_dir, '../models', pretrain_dataset, 'vgg16')
    output_dir = join(source_dir, 'ft_' + dataset_name)  # new folder for fine-tuned models

    # import requests
    #
    # print('Download Starting...')
    # url = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
    # r = requests.get(url)
    # with open(join(dataset_dir, 'pets.zip'), 'wb') as output_file:
    #     output_file.write(r.content)
    # print('Download Completed!!!')


    # # split into 80% train and 20% test folders
    # import splitfolders  # or import split_folders
    # # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    # splitfolders.ratio(join(dataset_dir, 'cell_images'), output=dataset_dir, seed=1, ratio=(.8, .2), group_prefix=None)



    ## class labels for pets dataset
    # {'Sphynx': 0, 'Russian_Blue': 1, 'keeshond': 2, 'Maine_Coon': 3, 'Bombay': 4, 'wheaten_terrier': 5,
    #  'Egyptian_Mau': 6, 'havanese': 7, 'yorkshire_terrier': 8, 'pomeranian': 9, 'shiba_inu': 10, 'Persian': 11,
    #  'japanese_chin': 12, 'beagle': 13, 'Abyssinian': 14, 'Siamese': 15, 'chihuahua': 16, 'basset_hound': 17,
    #  'american_pit_bull_terrier': 18, 'staffordshire_bull_terrier': 19, 'english_setter': 20, 'samoyed': 21,
    #  'american_bulldog': 22, 'Bengal': 23, 'Ragdoll': 24, 'British_Shorthair': 25, 'newfoundland': 26, 'boxer': 27,
    #  'Birman': 28, 'german_shorthaired': 29, 'scottish_terrier': 30, 'english_cocker_spaniel': 31, 'leonberger': 32,
    #  'miniature_pinscher': 33, 'pug': 34, 'saint_bernard': 35, 'great_pyrenees': 36}


if print_pets:

    dataset_name = 'pets'
    dataset_dir = join(os.getcwd(), '../data', dataset_name)
    bs = 6

    dataset = Pets(dataset_dir=dataset_dir, device=device)
    class_names = dataset.class_names
    print(class_names)
    train_loader = dataset.get_train_loader(batch_size=bs, shuffle=True)
    test_loader = dataset.get_test_loader(batch_size=bs)

    images, labels = next(iter(train_loader))
    print(images, labels)

    fig = plt.figure(figsize=(9, 7))
    for i in range(6):
        ax = fig.add_subplot(2, 3, i + 1, xticks=[], yticks=[], title=class_names[labels[i]])
        imshow(images[i])
    plt.show()
