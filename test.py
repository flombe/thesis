import numpy as np

Input = [np.array([1, 2, 3]),
		np.array([4, 5, 6]),
		np.array([7.2, 8.2, 9.2])]

print([[*(Input[m][k] for m in range(len(Input)))] for k in range(3)])
print([np.mean([*(Input[m][k] for m in range(len(Input)))]) for k in range(3)])

from os.path import join
check = ['erst', 'zweit', 'dre']
print([join('_'+check+'.pt') for check in check])

# import os
# import torch
# for file in os.listdir(''):
#     model_dict = torch.load(f'folder/folder/{file}')


# import torch
# # set device
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     print("Devise used = cuda on ", torch.cuda.get_device_name(device))
# else:
#     device = torch.device("cpu")
#     print("Devise used = ", device)

first_batches_chkpts = np.array([1, 3, 10, 30, 100, 300])
zwei = np.array([1,3,10,30,100])
out = first_batches_chkpts * 0.001
print(out)
print(np.append(out, zwei))


import pandas as pd
import json
import os
root_dir = os.getcwd()
file = '/mnt/antares_raid/home/bemmerl/thesis/models/mnist2class/models_1/pre_mnist2_train_stats.json'
with open(file, 'r') as myfile:
    data = myfile.read()
obj = json.loads(data)
df = pd.DataFrame([obj])
print('df: ', df)

dff = pd.DataFrame([obj]*5)

ddff = pd.DataFrame()

ddff = ddff.append(dff, ignore_index=True)

print('ddff: ', ddff)


ddff.insert(2, 'pre_dataset', 'mnist')

param = {'train_samples': 60000,
		 'batch_size': 64,
		 'lr': 0.0001}
ddff.insert(3, 'pre_param', [param]*len(ddff))

print(ddff['pre_param'][0])
print(ddff['pre_param'][0]["lr"])
