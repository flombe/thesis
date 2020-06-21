import numpy as np

Input = [np.array([1, 2, 3]),
		np.array([4, 5, 6]),
		np.array([7.2, 8.2, 9.2])]

print([[*(Input[m][k] for m in range(len(Input)))] for k in range(3)])
print([np.mean([*(Input[m][k] for m in range(len(Input)))]) for k in range(3)])

from os.path import join
check = ['erst', 'zweit', 'dre']
print([join('_'+check+'.pt') for check in check])

import os
import torch
for file in os.listdir(''):
    model_dict = torch.load(f'folder/folder/{file}')


import torch
# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)
