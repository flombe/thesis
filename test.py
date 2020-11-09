import numpy as np
import torch
import os
from os.path import join
import pandas as pd
import json

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)


root_dir = os.getcwd()
model_dir = join(root_dir, 'models', 'vgg16/random_init')
df = pd.DataFrame()

for seed in range(1, 4):
	df_path = join(model_dir, f'models_{seed}/df_pre_random_init+metrics')
	df_pre = pd.read_pickle(df_path)
	df = df.append(df_pre, ignore_index=True)

df.insert(2, 'seed', [1,2,3])
# group by name and get mean and std over the 3 seeds
df.to_pickle(join(model_dir, 'df_pre_random_init'))




