import numpy as np
import torch
import os
from os.path import join
import pandas as pd
import json
from rsa import get_rdm_metric
from rsa import get_rdm_metric_vgg

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)


dataset = 'vggface'
root_dir = os.getcwd()
model_dir = join(root_dir, 'models', 'vgg16', dataset)

path = join(model_dir, f'df_pre_{dataset}+metrics')
if os.path.exists(path):
    df = pd.read_pickle(path)
    print(df['RSA_custom3D'][0])
    rsa = get_rdm_metric_vgg(dataset, 'custom3D')
    print(rsa)
    df.at[0, 'RSA_custom3D'] = rsa
    df.to_pickle(path)
else:
    df = pd.DataFrame()
    for seed in range(1, 4):
        path = join(model_dir, f'models_{seed}/df_pre_{dataset}+metrics')
        df_pre = pd.read_pickle(path)
        print(df_pre['RSA_custom3D'][0])
        rsa = get_rdm_metric_vgg(join(dataset, f'models_{seed}'), 'custom3D')
        print(rsa)
        df_pre.at[0, 'RSA_custom3D'] = rsa
        print(df_pre['RSA_custom3D'][0])
        df = df.append(df_pre, ignore_index=True)
    df.to_pickle(join(model_dir, 'df_pre_random_init+metrics'))

print(f' > Done. df with new RSA metric saved for {dataset}')




