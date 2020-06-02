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


# plot ids
plt.figure(figsize=(7, 6), dpi=100)
for name, ss, ids in new_list:
    plt.plot(range(len(ids)), ids, label=name)
plt.xlabel("Layers")
plt.ylabel("Intrinsic Dimension")
plt.legend()
plt.show()

for name, ss, ids in new_list:
    plt.plot(range(len(ss)), ss, label=name)
plt.xlabel("Layers")
plt.ylabel("SSW/TSS")
plt.legend()
plt.show()






