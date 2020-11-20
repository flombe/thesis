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


dataset_name = 'vgg16/malaria'

root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)
model_dir = join(root_dir, 'models', dataset_name)

# to add multiple seed runs (e.g. on random_init) into one df
df = pd.DataFrame()
for seed in range(1, 10):
    dff = pd.read_json(f'/mnt/antares_raid/home/bemmerl/thesis/models/vgg16/malaria/models_{seed}/pre_malaria_train_stats.json')
    dff['seed'] = seed
    df = df.append(dff, ignore_index=True)
df.to_pickle(join(model_dir, f'df_pre_malaria'))


