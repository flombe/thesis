import os
from os.path import join
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd

import mnist_archs
import vgg_arch
import train_utils
import datasets


# parse args from sh script
parser = train_utils.train_args_parser()
args = parser.parse_args()
dataset_name, batch_size, epochs, lr, run_name, seeds = train_utils.parse_train_args(args)
print('>> Run {run_name} on dataset {dataset} for {seeds} different seeds. <<'.format(
    run_name=run_name, dataset=dataset_name, seeds=seeds))

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)

# set directory
root_dir = os.getcwd()
dataset_dir = join(root_dir, '../data', dataset_name)
model_dir = join(root_dir, '../models', dataset_name)
os.makedirs(model_dir, exist_ok=True)

# save training stats in df
dff = pd.DataFrame()

# run every training nr. of seeds times to aggregate results for statistical testing
for seed_run in range(1, seeds+1):
    # set seed
    train_utils.set_seed(seed_run)

    if dataset_name == 'mnist':
        model = mnist_archs.mnistConvNet()
        dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)

    elif dataset_name == 'mnist2class':
        model = mnist_archs.mnistConvNet2class()
        dataset = datasets.MNIST2class(dataset_dir=dataset_dir, device=device)

    elif dataset_name == 'mnist_noise':
        model = mnist_archs.mnistConvNet()
        dataset = datasets.MNIST_noise(dataset_dir=dataset_dir, device=device)

    elif dataset_name == 'mnist_split1':
        model = mnist_archs.mnistConvNet5class()
        dataset = datasets.MNIST_split1(dataset_dir=dataset_dir, device=device)

    elif dataset_name == 'mnist_split2':
        model = mnist_archs.mnistConvNet5class()
        dataset = datasets.MNIST_split2(dataset_dir=dataset_dir, device=device)

    elif dataset_name == 'mnist_noise_struct':
        model = mnist_archs.mnistConvNet()
        dataset = datasets.MNIST_noise_struct(dataset_dir=dataset_dir, device=device)

    elif dataset_name == 'fashionmnist':
        model = mnist_archs.mnistConvNet()
        dataset = datasets.FashionMNIST(dataset_dir=dataset_dir, device=device)

    elif dataset_name == 'cifar10':
        model = vgg_arch.vgg16(pretrained=False, num_classes=10)
        dataset = datasets.CIFAR10(dataset_dir=dataset_dir, device=device)

    elif dataset_name == 'custom3D':
        model = vgg_arch.vgg16(pretrained=False, num_classes=40)
        dataset = datasets.Custom3D(dataset_dir=dataset_dir, device=device)

    elif dataset_name == 'malaria':
        model = vgg_arch.vgg16(pretrained=False, num_classes=2)
        dataset = datasets.Malaria(dataset_dir=dataset_dir, device=device)

    elif dataset_name == 'pets':
        model = vgg_arch.vgg16(pretrained=False, num_classes=37)
        dataset = datasets.Pets(dataset_dir=dataset_dir, device=device)

    model.to(device)
    criterion = F.nll_loss  # with F.log_softmax on output = CrossEntropyLoss
    if dataset_name in ['cifar10', 'custom3D', 'malaria', 'pets']: criterion = F.cross_entropy
    print(model)

    # loaders
    train_loader = dataset.get_train_loader(batch_size=batch_size)
    test_loader = dataset.get_test_loader(batch_size=batch_size)

    # Training
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = False
    if dataset_name == 'cifar10':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if dataset_name in ['custom3D', 'malaria', 'pets']:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_acc, train_loss, test_acc, test_loss, df = train_utils.train(model=model, train_loader=train_loader,
                                                                       test_loader=test_loader, optimizer=optimizer,
                                                                       device=device, criterion=criterion, epochs=epochs,
                                                                       output_dir=model_dir, run_name=run_name,
                                                                       seed=seed_run, save=True, scheduler=scheduler)
    dff = dff.append(df, ignore_index=True)
    print('Done trainings run ', seed_run)

# add params to dff and save
dff.insert(3, 'pre_dataset', dataset_name)
param = {'train_samples': len(train_loader)*batch_size,
         'batch_size': batch_size,
         'lr': lr}
dff.insert(4, 'pre_param', [param] * len(dff))
dff.to_pickle(join(model_dir, 'df_' + run_name))

print('Done with all training runs.')
