import numpy as np
import torch
import os
from os.path import join
import pandas as pd
import json


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)


trained_dataset = 'mnist'
target_dataset = 'custom3D'

root_dir = os.getcwd()
models_dir = join(root_dir, 'models', trained_dataset)

# load df
df_path = join(models_dir, f'df_pre_{trained_dataset}+metrics')
#df_path = join(models_dir, f'ft_malaria/df_ft_random_init_malaria')
df_path = '/mnt/antares_raid/home/bemmerl/thesis/models/vgg16/cars/df_pre_cars+metrics'
df = pd.read_pickle(df_path)




# seed = 1
# name = 'model_pre_mnist_0_1.pt'
# print(df.loc[df.seed.eq(seed) & df.model_name.eq(name), ['ID_noise']])
# df.at[0, 'ID_target'] = 1


# ft_dir = join(models_dir, 'ft_mnist')
#
# df_all = pd.DataFrame()
# for seed in range(1, 10):  # only till 9, 10 already in df
#     #json_data = []
#     folder = join('models_' + str(seed))
#     print(folder)
#     for file in os.listdir(join(ft_dir, folder)):
#         if file.endswith(".json"):
#             print(file)
#             dff = pd.read_json(join(ft_dir, folder, file))
#             df_all = df_all.append(dff, ignore_index=True)
#
# df_all.insert(3, 'ft_dataset', 'mnist')
# df_all.insert(4, 'ft_param', [df['ft_param'][1]] * len(df_all))
#
# df_ft = df_all.append(df, ignore_index=True)

### df_ft.to_pickle(df_path)
