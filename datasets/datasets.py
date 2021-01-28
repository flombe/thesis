"""
Classes for defining dataloader and transformation for each dataset
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from torchvision import datasets, transforms
import numpy as np
import os
from glob import glob


class Dataset:
    def get_test_transform(self):
        raise NotImplementedError()

    def get_train_transform(self):
        return self.get_test_transform()

    def get_test_loader(self, batch_size=32):
        pass

    def get_train_loader(self, batch_size=32):
        pass

    def name(self):
        raise NotImplementedError()


class TorchDataset(Dataset):
    def __init__(self, dataset_dir, device):
        self.dataset_dir = dataset_dir
        self.loader_args = {'num_workers': 10, 'pin_memory': True} if device.type == 'cuda' else {}

    def get_train_loader(self, batch_size=32):
        train_loader = torch.utils.data.DataLoader(
            self.get_dataset_cls()(self.dataset_dir, train=True, download=True,
                                   transform=self.get_train_transform()), batch_size=batch_size, shuffle=True,
            **self.loader_args)
        return train_loader

    def get_test_loader(self, batch_size=32, shuffle=True):
        test_loader = torch.utils.data.DataLoader(
            self.get_dataset_cls()(self.dataset_dir, train=False,
                                   transform=self.get_test_transform()), batch_size=batch_size, shuffle=shuffle,
            **self.loader_args)
        return test_loader

    def get_dataset_cls(self):
        raise NotImplementedError()


class MNIST(TorchDataset):
    def get_test_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return transform

    def get_dataset_cls(self):
        return datasets.MNIST

    def name(self):
        return 'mnist'


class MNIST2class(MNIST):
    classes = ['0 - even', '1 - uneven']

    def name(self):
        return 'mnist2class'

    def get_train_loader(self, batch_size=32):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.dataset_dir, train=True, download=False,  # use MNIST loader since same data
                           transform=self.get_train_transform()),
            batch_size=batch_size, shuffle=True,
            **self.loader_args)
        return train_loader

    def get_test_loader(self, batch_size=32, shuffle=True):
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.dataset_dir, train=False,
                           transform=self.get_test_transform()), batch_size=batch_size, shuffle=shuffle,
            **self.loader_args)
        return test_loader


class MNIST_split1(MNIST2class):
    def name(self):
        return 'mnist_split1'


class MNIST_split2(MNIST2class):
    def name(self):
        return 'mnist_split2'


class MNIST_noise_struct(MNIST):
    def name(self):
        return 'mnist_noise_struct'

    def get_train_loader(self, batch_size=32):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.dataset_dir, train=True, download=False,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True,
            **self.loader_args)
        return train_loader

    def get_test_loader(self, batch_size=32, shuffle=True):
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(self.dataset_dir, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=shuffle,
            **self.loader_args)
        return test_loader


class MNIST_noise(MNIST_noise_struct):
    def name(self):
        return 'mnist_noise'


class FashionMNIST(TorchDataset):
    def get_test_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        return transform

    def get_dataset_cls(self):
        return datasets.FashionMNIST

    def name(self):
        return 'fashionmnist'


class CIFAR10(TorchDataset):
    def get_train_transform(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform

    def get_test_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform

    def get_dataset_cls(self):
        return datasets.CIFAR10

    def name(self):
        return 'cifar10'


class Custom3D(Dataset):
    def __init__(self, dataset_dir, device):
        self.dataset_dir = dataset_dir
        self.loader_args = {'num_workers': 10, 'pin_memory': True} if device.type == 'cuda' else {}

        self.train_data = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), self.get_train_transform())
        self.test_data = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), self.get_test_transform())
        self.class_names = self.train_data.classes

    def __len__(self):
        return len(self.train_data)+len(self.test_data)

    def __getitem__(self, idx):
        if idx < len(self.train_data):
            image, label = self.train_data[idx]
        else:
            image, label = self.test_data[idx - len(self.train_data)]
        return image, self.class_names[label]

    def get_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform

    def get_test_transform(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform

    def get_train_loader(self, batch_size=32, shuffle=True):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=shuffle, **self.loader_args)
        return train_loader

    def get_test_loader(self, batch_size=32, shuffle=True):
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, shuffle=shuffle, **self.loader_args)
        return test_loader

    def get_dataset_cls(self):
        return datasets.Custom3D

    def name(self):
        return 'custom3D'


class Malaria(Dataset):
    def __init__(self, dataset_dir, device):
        self.dataset_dir = dataset_dir
        self.loader_args = {'num_workers': 10, 'pin_memory': True} if device.type == 'cuda' else {}

        self.train_data = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), self.get_train_transform())
        self.test_data = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), self.get_test_transform())
        self.class_names = self.train_data.classes

    def __len__(self):
        return len(self.train_data)+len(self.test_data)

    def __getitem__(self, idx):
        if idx < len(self.train_data):
            image, label = self.train_data[idx]
        else:
            image, label = self.test_data[idx - len(self.train_data)]
        return image, self.class_names[label]

    def get_train_transform(self):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # with crop, cause sample dims are varying
            transforms.ColorJitter(0.05),  # how much to jitter brightness (first arg)
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # since values between -1, 1
        ])
        return transform

    def get_test_transform(self):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return transform

    def get_train_loader(self, batch_size=32, shuffle=True):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=shuffle, **self.loader_args)
        return train_loader

    def get_test_loader(self, batch_size=32, shuffle=True):
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, shuffle=shuffle, **self.loader_args)
        return test_loader

    def get_dataset_cls(self):
        return datasets.Malaria

    def name(self):
        return 'malaria'


class Pets(Dataset):  # pretty hacky mess - but works
    # https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    def __init__(self, dataset_dir=None, device=None, transform=None):
        self.dataset_dir = dataset_dir
        self.transforms = transform
        self.data = []

        if dataset_dir and device:
            self.loader_args = {'num_workers': 10, 'pin_memory': True} if device.type == 'cuda' else {}
            self.data, self.class_names = self.get_data()

            self.train_data, self.test_data = self.data_split(test_split=0.2)


    def __getitem__(self, index):
        img, label = self.data[index]

        if self.transforms:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data)

    def get_train_loader(self, batch_size=32, shuffle=True):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=shuffle,
                                                   **self.loader_args)
        return train_loader

    def get_test_loader(self, batch_size=32, shuffle=True):
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, shuffle=shuffle,
                                                  **self.loader_args)
        return test_loader

    def get_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

    def get_test_transform(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

    def get_data(self):
        filenames = glob(os.path.join(self.dataset_dir, 'images/*.jpg'))

        classes = set()
        data = []
        labels = []

        # Load the images and get the classnames from the image path
        from PIL import Image
        for image in filenames:
            class_name = image.rsplit("/", 1)[1].rsplit('_', 1)[0]
            classes.add(class_name)
            img = Image.open(image).convert('RGB')

            data.append(img)
            labels.append(class_name)

        # convert classnames to indices
        class2idx = {cl: idx for idx, cl in enumerate(classes)}
        labels = torch.Tensor(list(map(lambda x: class2idx[x], labels))).long()
        data = list(zip(data, labels))
        return data, list(classes)

    def data_split(self, test_split):
        class_values = [[] for x in range(len(self.class_names))]

        # create arrays for each class type
        for d in self.data:
            class_values[d[1].item()].append(d)

        train_data = Pets(transform=self.get_train_transform())
        test_data = Pets(transform=self.get_test_transform())

        # put (1-test_split) of the images of each class into the train dataset
        # and test_split of the images into the test dataset
        for class_dp in class_values:
            split_idx = int(len(class_dp) * (1 - test_split))
            train_data.data += class_dp[:split_idx]
            test_data.data += class_dp[split_idx:]

        return train_data, test_data

    def get_dataset_cls(self):
        return datasets.Pets

    def name(self):
        return 'pets'


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
