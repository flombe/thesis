import torch
import datasets
import os
from os.path import join
import numpy as np
import json
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# root_dir = os.getcwd()
# dataset_dir = join(root_dir, 'data', dataset_name)
# print('Dataset directory -- ' + dataset_dir)
# model_dir = join(dataset_dir, 'models')



file = join('/mnt/antares_raid/home/bemmerl/thesis/data/mnist2class/models_1', 'ss_id.json')
with open(file, 'r') as myfile:
    data = myfile.read()
new_list = json.loads(data)['model_pre_mnist2']
print(new_list)

xticks = ['in', 'conv1', 'pool1', 'conv2','pool2', 'fc1', 'output']
# plot ids
plt.style.use('seaborn')

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20c.colors)
fig, axs = plt.subplots(2, sharex=True, figsize=(7, 9), dpi=150)
axs[0].set_title('SumSqr and ID over model layers')

for name, ss, ids in new_list:
    axs[0].plot(range(len(ids)), ids, '.-')
axs[0].set_ylabel("Intrinsic Dimension")

for name, ss, ids in new_list:
    axs[1].plot(range(len(ss)), ss,'.-', label=name)
plt.xlabel("Layers")
plt.xticks(range(7), labels=xticks)
plt.ylabel("SSW/TSS")
plt.legend(loc="lower left", prop={'size': 9})
plt.show()







