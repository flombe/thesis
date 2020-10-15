import torch
import os
from os.path import join
import numpy as np
import train_utils
import shutil

###
create_dataset = 'mnist_split2'  # 'mnist_split1'


# set seed
train_utils.set_seed(1)

# Split MNIST dataset into mnist_split1 and mnist_split2
dataset_name = 'mnist'
root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)

# augment values of MNIST dataset pictures - random noise
train_data, train_labels = torch.load(join(dataset_dir, 'processed/training.pt'))
print(np.shape(train_data), np.shape(train_labels))
test_data, test_labels = torch.load(join(dataset_dir, 'processed/test.pt'))
print(np.shape(test_data), np.shape(test_labels))


# new dir for new mnist_noise dataset
new_dir = join(root_dir, 'data', create_dataset)
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

save_dir = join(new_dir, 'processed')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('Mnist Split data results directory: ' + save_dir)

train_labels_split = []
train_data_split = []
test_labels_split = []
test_data_split = []

if create_dataset == 'mnist_split1':
    labels = [0, 1, 2, 3, 4]
else: labels = [5, 6, 7, 8, 9]

for label in labels:
    idx = train_labels == label
    print(idx)
    print(len(train_labels[idx]))
    train_labels_split.append(train_labels[idx])
    train_data_split.append(train_data[idx])
    print(len(train_labels_split[label-5]))

    idxx = test_labels == label
    test_labels_split.append(test_labels[idxx])
    test_data_split.append(test_data[idxx])
    print(len(test_labels_split[label-5]))

train_labels_split = torch.cat(train_labels_split)
train_data_split = torch.cat(train_data_split)
test_labels_split = torch.cat(test_labels_split)
test_data_split = torch.cat(test_data_split)

print(train_labels_split.shape, train_data_split.shape)
print(test_labels_split.shape, test_data_split.shape)

# adj labels for mnist_split2 to fit NN output and to use same implementation as for other datasets
if create_dataset == 'mnist_split2':
    print(train_labels_split)
    train_labels_split = train_labels_split-5
    print(train_labels_split)
    test_labels_split = test_labels_split - 5

# save as torch pt file
torch.save([train_data_split, train_labels_split], 'training.pt')
torch.save([test_data_split, test_labels_split], 'test.pt')

# move created files
shutil.move('training.pt', save_dir)
shutil.move('test.pt', save_dir)



## mnist not balanced?

# mnist_split1
# train : torch.Size([30596]) torch.Size([30596, 28, 28])
# test : torch.Size([5139]) torch.Size([5139, 28, 28])

# mnist_split2
# train : torch.Size([29404]) torch.Size([29404, 28, 28])
# test : torch.Size([4861]) torch.Size([4861, 28, 28])

# because mnist torch dataset (training):
# labels[0 1 2 3 4 5 6 7 8 9]
# count [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]
