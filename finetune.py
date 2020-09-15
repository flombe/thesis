import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from os.path import join
import pandas as pd

import datasets
import train_utils
import mnist_archs


# parse args from sh script
parser = train_utils.train_args_parser()
parser.add_argument('--pre_dataset')  # add parser arg for pre-trained selection
args = parser.parse_args()
dataset_name, batch_size, epochs, lr, run_name, seeds = train_utils.parse_train_args(args)
pretrain_dataset = args.pre_dataset
print(' >> Run {run_name} on dataset {dataset} on pre-trained {pre} models. <<'.format(
    run_name=run_name, dataset=dataset_name, pre=pretrain_dataset))

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)

# set dir
root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)  # target data for ft
source_dir = join(root_dir, 'models', pretrain_dataset)
output_dir = join(source_dir, 'ft_' + dataset_name)  # new folder for fine-tuned models

# save training stats in df
dff = pd.DataFrame()

for seed in range(1, 11):
    model_dir = join(source_dir, 'models_' + str(seed))
    print(model_dir)

    for file in os.listdir(model_dir):
        if file.startswith("model_"):
            print(file)
            # Load model
            model = torch.load(join(model_dir, file))

            # add pre-train checkpoint name to run_name for ft
            # print(file.split(str(pretrain_dataset), 1)[1][:-3])  # add source model name for saving
            pretrain_checkpt = file.split(str(pretrain_dataset), 1)[1][:-3]  # naming is eg. model_pre_mnist_0_1.pt
            run_name_sub = join(run_name + pretrain_checkpt)

            # print(list(model.fc1.parameters()))
            # new fine-tune fc layers
            model_ft = model
            model_ft.fc1 = mnist_archs.mnistConvNet().fc1  # = nn.Linear(1600, 128)
            model_ft.fc2 = mnist_archs.mnistConvNet().fc2  # = nn.Linear(128, 10)
            # print(model_ft.fc1.parameters())
            print('fc layers newly initialized')

            model_ft = model_ft.to(device)

            # from train - dataset and criterion selection
            if dataset_name == 'mnist':
                dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)
            elif dataset_name == 'fashionmnist':
                dataset = datasets.FashionMNIST(dataset_dir=dataset_dir, device=device)
            elif dataset_name == 'mnist_noise_struct':
                dataset = datasets.MNIST_noise_struct(dataset_dir=dataset_dir, device=device)
            criterion = F.nll_loss
            ## for 'cifar10' diff. layers and criterion (F.cross_entropy)

            # loaders
            train_loader = dataset.get_train_loader(batch_size=batch_size)
            test_loader = dataset.get_test_loader(batch_size=batch_size)

            # Training
            optimizer = optim.Adam(model_ft.parameters(), lr=lr)
            train_acc, train_loss, test_acc, test_loss, df = train_utils.train(model=model_ft, train_loader=train_loader,
                                                                           test_loader=test_loader, optimizer=optimizer,
                                                                           device=device, criterion=criterion,
                                                                           epochs=epochs, output_dir=output_dir,
                                                                           run_name=run_name_sub, seed=seed, ft=True)
            dff = dff.append(df, ignore_index=True)
            print(dff)
            print('Done fine-tuning run ', model)

    print('Done all models of seed ', seed)

# add to dff and save
dff.insert(3, 'ft_dataset', dataset_name)
param = {'train_samples': len(train_loader)*batch_size,
         'batch_size': batch_size,
         'lr': lr}
dff.insert(4, 'ft_param', [param] * len(dff))
dff.to_pickle(join(output_dir, 'df_' + run_name))

print('Done fine-tuning all models for all seeds.')
