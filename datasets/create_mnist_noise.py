import torch
import os
from os.path import join
import shutil
import numpy as np
from train import train_utils

# set seed
train_utils.set_seed(1)

# Augment MNIST dataset to pure random noise
dataset_name = 'mnist'
root_dir = os.getcwd()
dataset_dir = join(root_dir, '../data', dataset_name)

# augment values of MNIST dataset pictures - random noise
train_data, train_labels = torch.load(join(dataset_dir, 'processed/training.pt'))
print(np.shape(train_data))
train_data = np.random.uniform(0, 1, (60000, 784))

test_data, test_labels = torch.load(join(dataset_dir, 'processed/test.pt'))
print(np.shape(test_data))
test_data = np.random.uniform(0, 1, (10000, 784))

# train
train_data = train_data.reshape((-1, 28, 28))
train_data = torch.from_numpy(train_data)
print(type(train_data), train_data.shape)
# save as torch pt file
torch.save([train_data, train_labels], 'training.pt')

# test
test_data = test_data.reshape((-1, 28, 28))
test_data = torch.from_numpy(test_data)
print(type(test_data), test_data.shape)
# save as torch pt file
torch.save([test_data, test_labels], 'test.pt')


# new dir for new mnist_noise dataset
new_dir = join(root_dir, '../data', 'mnist_noise')
# check if directory exists or not yet
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

save_dir = join(new_dir, 'processed')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('Noise_structured data results directory: ' + save_dir)

# move created files
shutil.move('training.pt', save_dir)
shutil.move('test.pt', save_dir)
