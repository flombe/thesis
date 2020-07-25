import numpy as np

Input = [np.array([1, 2, 3]),
		np.array([4, 5, 6]),
		np.array([7.2, 8.2, 9.2])]

print([[*(Input[m][k] for m in range(len(Input)))] for k in range(3)])
print([np.mean([*(Input[m][k] for m in range(len(Input)))]) for k in range(3)])

from os.path import join
check = ['erst', 'zweit', 'dre']
print([join('_'+check+'.pt') for check in check])

first_batches_chkpts = np.array([1, 3, 10, 30, 100, 300])
zwei = np.array([1,3,10,30,100])
out = first_batches_chkpts * 0.001
print(out)
print(np.append(out, zwei))


# dataframe handling
import pandas as pd
import json
import os
root_dir = os.getcwd()
model_dir = join(root_dir, 'models', 'mnist2class')
dff = pd.DataFrame()
for i in range(1,11):
	print('read models_', i)
	file = join(model_dir, 'models_' + str(i), 'pre_mnist2_train_stats.json')
	with open(file, 'r') as myfile:
		data = myfile.read()
	obj = json.loads(data)
	df = pd.DataFrame(obj)
	dff = dff.append(df, ignore_index=True)


##############
import torch
# check if extract balanced
dataset_trained = 'mnist'
dataset_extracted = 'fashionmnist'

for seed in range(1, 11):
	root_path = os.getcwd()
	models_dir = join(root_path, 'models', dataset_trained, 'models_' + str(seed))

	load_extracted = join(models_dir, dataset_extracted + '_extracted.pt')
	models = torch.load(load_extracted)
	print(f'loaded {len(models)} models from - {load_extracted}')

	# get labels once (since constant) for plotting
	labels = list(models.values())[0]['labels']  # tensor, same for all models - use later for plots
	print('labels and count: ', np.unique(np.array(labels), return_counts=True))  # check if balanced data
##############


import torch
# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)

import datasets
dataset_name = 'mnist' ##
dataset_dir = join(root_dir, 'data', dataset_name)
batch_size = 64
lr = 0.0001
run_name = 'test' ##
dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)
# train_loader = dataset.get_train_loader(batch_size=batch_size)

# print('train_loader len * bs: ', len(train_loader)*batch_size)
# print('len train_loader.dataset; ', len(train_loader.dataset))




# sort dir files with natsort
models_dir = '/mnt/antares_raid/home/bemmerl/thesis/models/mnist2class'
all_models = {}
index = 0
from natsort import natsorted
for i in range(1, 11):
	model_dir = join(models_dir, 'models_' + str(i))
	for file in natsorted(os.listdir(model_dir)):
		if file.endswith(".pt"):
			print(index, file)
			index += 1
print('Extracted all 10 model folders.')



print('---- Extracted Test ----')
import torch

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)

file = '/mnt/antares_raid/home/bemmerl/thesis/models/mnist/models_1/mnist_extracted.pt'
data = torch.load(file)
print(type(data))
print(len(data))
print(data.keys())

keys = list(data.keys())
print("Key : {} , Value type: {} ,  Value dim : {}".format(keys[0], type(data[keys[0]]), len(data[keys[0]])))

print('Keys of value dict: ', data[keys[0]].keys())
for i in range(7):
	print('Len of activations for one model: ', list(data[keys[0]]['layers'])[i].shape)

# act = pd.DataFrame.from_dict(data)

### handy string manipulation
# overlapping dict keys (model names) - add dataset names
# corr_dict_source = {f'{k}@{dataset_trained}': v for k, v in corr_dict_source.items()}
# corr_dict_target = {f'{k}@{dataset_extracted}': v for k, v in corr_dict_target.items()}
