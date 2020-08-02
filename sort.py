import torch
import datasets
import os
from os.path import join
import numpy as np
import pandas as pd
from natsort import natsorted


# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)


dataset_name = 'mnist'
run_name = 'pre_mnist'

root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)
model_dir = join(root_dir, 'models', dataset_name)

df_added = pd.read_pickle(join(model_dir, 'df_pre_mnistadded'))

for i in range(0, 119, 12):
    print(i)
    df_added[i:i+12] = df_added[i:i+12].sort_values(by=['model_name'], ignore_index=True)

# df = df_added
# df['model_name'] = pd.Categorical(df['model_name'], ordered=True, categories=natsorted(df['model_name'].unique()))
# df = df.sort_values('model_name')

# df.to_pickle(join(model_dir, 'df_' + run_name + 'added'))
