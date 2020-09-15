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


## sort by seed and model_name using natsort
# df = df_added
# idx, *_ = zip(*natsorted(zip(df.index, df.seed, df.model_name), key=lambda x: (x[1], x[2])))
# df = df.iloc[list(idx)]

# df.to_pickle(join(model_dir, 'df_' + run_name + 'added'))
