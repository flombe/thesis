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

for seed in range(1,4):
	df_path = join(model_dir, f'ft_custom3D/models_{seed}/df_ft_random_init_custom3D')
	df_ft = pd.read_pickle(df_path)
	print(df_ft)
	df = df.append(df_ft, ignore_index=True)

# group by name and get mean and std over the 3 seeds

