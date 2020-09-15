import torch
import os
from os.path import join
import shutil
import numpy as np
import train_utils

# set seed
train_utils.set_seed(1)

# Augment MNIST dataset to random noise, with a bit of structure for the different labels
dataset_name = 'mnist'
root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)

# augment values of MNIST dataset pictures - random noise
train_data, train_labels = torch.load(join(dataset_dir, 'processed/training.pt'))
print(np.shape(train_data))
train_data = np.random.uniform(0, 1, (60000, 784))

test_data, test_labels = torch.load(join(dataset_dir, 'processed/test.pt'))
print(np.shape(test_data))
test_data = np.random.uniform(0, 1, (10000, 784))

# percentage of values that share the same value per label
ratio = 0.1

for label in range(0, 10):
    idx = np.random.choice(784, int(784 * ratio), replace=False)  # get 78 random indices
    label_values = np.ones(int(784 * ratio)) * label/10  # assign same 0.1 * label nr. value
    # label_values = np.random.uniform(0, 1, int(784 * rate_fixed))  # problem: if some are randomly very close together
    for i in range(len(train_labels)):
        if train_labels[i] == label:
            train_data[i][idx] = label_values  # change 78 values of these labels to a shared value
    for i in range(len(test_labels)):
        if test_labels[i] == label:
            test_data[i][idx] = label_values


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


# new dir for new MNIST2class dataset
new_dir = join(root_dir, 'data', 'mnist_noise_struct')
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
