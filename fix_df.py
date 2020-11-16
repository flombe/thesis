import torch
import datasets
import os
from os.path import join
import numpy as np
import json
from pathlib import Path
import train_utils
import mnist_archs
import pandas as pd


# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)

dataset_name = 'mnist'

root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)
model_dir = join(root_dir, 'models', dataset_name)

df = pd.read_pickle(join(model_dir, f'ft_fashionmnist/df_ft_{dataset_name}_fashion'))

for seed in range(1, 11):
    dff = pd.read_json(f'/mnt/antares_raid/home/bemmerl/thesis/models/mnist/ft_fashionmnist/models_{seed}/ft_mnist_fashion_0_train_stats.json')
    df = df.append(dff, ignore_index=True)




# df = pd.read_pickle(join(model_dir, f'df_pre_{dataset_name}+metrics'))
# print(df.columns)
# df.columns = ['model_name', 'seed', 'pre_net', 'pre_dataset', 'pre_param', 'pre_epochs',
#        'pre_train_acc', 'pre_train_loss', 'pre_test_acc', 'pre_test_loss',
#        'ID_mnist', 'SS_mnist', 'RSA_mnist']
# print(df.columns)
# df.to_pickle(join(model_dir, f'df_pre_{dataset_name}+metrics'))


# run_name = 'pre_mnist'
# dataset = 'mnist'
# epochs = 100
# batch_size = 64
# lr = 0.0001
#
#
# first_batches_chkpts = np.array([0, 1, 3, 10, 30, 100, 300])
# epoch_chkpts = np.array([1, 3, 10, 30, 100])
#
# train_loss = []
# train_acc = []
# test_loss = []
# test_acc = []
# model_names = []
#
# for seed in range(1, 11):
#     model_seed_dir = join(model_dir, 'models_' + str(seed))
#     # set seed
#     train_utils.set_seed(seed)
#
#     if dataset_name == 'mnist':
#         model = mnist_archs.mnistConvNet()
#         dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)
#
#     model.to(device)
#
#     model_names.append(join('model_' + str(run_name) + '_0.pt'))
#     torch.save(model, join(model_seed_dir, model_names[-1]))
#
#     # loaders
#     train_loader = dataset.get_train_loader(batch_size=batch_size)
#     test_loader = dataset.get_test_loader(batch_size=batch_size)
#
#     # save train stats
#     model.eval()
#
#     loss0, acc0 = train_utils.evaluate(model, train_loader, device)
#     train_loss.append(loss0.item())
#     train_acc.append(acc0)
#
#     loss0, acc0 = train_utils.evaluate(model, test_loader, device)
#     test_loss.append(loss0.item())
#     test_acc.append(acc0)
#
#
# train_stats = {
#             'model_name': model_names,
#             'seed': range(1, 11),
#             'pre_net': model.__class__.__name__,
#             'pre_epochs': 0,
#             'pre_train_acc': train_acc,
#             'pre_train_loss': train_loss,
#             'pre_test_acc': test_acc,
#             'pre_test_loss': test_loss
#         }
# dff = pd.DataFrame(train_stats)
# df_added = df.append(dff, ignore_index=True)
# df_added.sort_values(by=['seed', 'model_name'], ignore_index=True)
#
# df_added.to_pickle(join(model_dir, 'df_' + run_name + 'added'))
