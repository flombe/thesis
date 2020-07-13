from torchvision import datasets, transforms
import torch
import os


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
            datasets.MNIST(self.dataset_dir, train=True, download=False,   ## use MNIST loader since same data
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
    def get_test_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return transform

    def get_dataset_cls(self):
        return datasets.CIFAR10

    def name(self):
        return 'cifar10'
