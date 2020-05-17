import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from os.path import join
import mnist_archs
import train_utils
import datasets
import vgg_mod

parser = train_utils.train_args_parser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10'])
args = parser.parse_args()
batch_size, epochs, lr, momentum = train_utils.parse_train_args(args)
dataset_name = args.dataset

root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)
os.makedirs(dataset_dir, exist_ok=True)
print('Results directory ' + dataset_dir)

if dataset_name == 'mnist':
    # model instantiation
    model = mnist_archs.Net()
    model.to(device)
    dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)
    criterion = F.nll_loss
else:
    model = vgg_mod.vgg16(pretrained=False, num_classes=10)
    model.to(device)
    dataset = datasets.CIFAR10(dataset_dir=dataset_dir, device=device)
    criterion = F.cross_entropy

# loaders
train_loader = dataset.get_train_loader(batch_size=batch_size)
test_loader = dataset.get_test_loader(batch_size=batch_size)

# Training
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
train_acc, train_loss, test_acc, test_loss = train_utils.train(model=model, train_loader=train_loader,
                                                               test_loader=test_loader, optimizer=optimizer,
                                                               device=device, criterion=criterion, epochs=epochs,
                                                               output_dir=dataset_dir)

print('Done.')
