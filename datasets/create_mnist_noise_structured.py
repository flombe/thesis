"""
Create mnist_noise_struct dataset by augmenting MNIST dataset to random noise with 10% structured data

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

# new dir for new mnist_noise_struct dataset
save_dir = join(root_dir, '../data', 'mnist_noise_struct', 'processed')
# check if directory exists or not yet
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# print('Noise_structured data results directory: ' + save_dir)


# augment values of MNIST dataset pictures - random noise
train_data, train_labels = torch.load(join(dataset_dir, 'processed/training.pt'))
print(np.shape(train_data))
train_data = np.random.uniform(0, 1, (60000, 784))

test_data, test_labels = torch.load(join(dataset_dir, 'processed/test.pt'))
print(np.shape(test_data))
test_data = np.random.uniform(0, 1, (10000, 784))

# percentage of values that should share the same value per label
ratio = 0.1

for label in range(0, 10):
    # select 78 random indices (for 10% of pixel value)
    idx = np.random.choice(784, int(784 * ratio), replace=False)
    # same distribution assigned to all label values
    label_values = np.random.uniform(0, 1, int(784 * ratio))
    for i in range(len(train_labels)):
        if train_labels[i] == label:
            # change 78 values of these labels to a shared value
            train_data[i][idx] = label_values
    for i in range(len(test_labels)):
        if test_labels[i] == label:
            test_data[i][idx] = label_values


# reshape into train sample size, convert to tensor and save
train_data = train_data.reshape((-1, 28, 28))
train_data = torch.from_numpy(train_data)
# print(type(train_data), train_data.shape)
torch.save([train_data, train_labels], join(save_dir, 'training.pt'))

# reshape into test sample size, convert to tensor and save
test_data = test_data.reshape((-1, 28, 28))
test_data = torch.from_numpy(test_data)
# print(type(test_data), test_data.shape)
torch.save([test_data, test_labels], join(save_dir, 'test.pt'))
