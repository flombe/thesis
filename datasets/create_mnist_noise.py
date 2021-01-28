"""
Create mnist_noise dataset by augmenting MNIST dataset to pure random noise

Note: MNIST dataset in /data/mnist needed
"""

import torch
import os
from os.path import join
import numpy as np
from train import train_utils

# set seed
train_utils.set_seed(1)

# set directories
dataset_name = 'mnist'
root_dir = os.getcwd()
dataset_dir = join(root_dir, '../data', dataset_name)

# new dir for new mnist_noise dataset
save_dir = join(root_dir, '../data', 'mnist_noise', 'processed')
# check if directory exists or not yet
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# print('Noise_structured data results directory: ' + save_dir)


# augment values of MNIST dataset pictures - random noise
train_data, train_labels = torch.load(join(dataset_dir, 'processed/training.pt'))
print(np.shape(train_data))
# overwrite train_data with random values to create noise samples
train_data = np.random.uniform(0, 1, (60000, 784))

# same for test with different shape
test_data, test_labels = torch.load(join(dataset_dir, 'processed/test.pt'))
print(np.shape(test_data))
test_data = np.random.uniform(0, 1, (10000, 784))


# reshape into train sample size, convert to tensor and save
train_data = train_data.reshape((-1, 28, 28))
train_data = torch.from_numpy(train_data)
print(type(train_data), train_data.shape)
torch.save([train_data, train_labels], join(save_dir, 'training.pt'))

# reshape into test sample size, convert to tensor and save
test_data = test_data.reshape((-1, 28, 28))
test_data = torch.from_numpy(test_data)
print(type(test_data), test_data.shape)
torch.save([test_data, test_labels], join(save_dir, 'test.pt'))
