import torch
import os
from os.path import join
import shutil
import numpy as np

# Augment MNIST dataset to only two classes (labels: "even"/0, "uneven"/1)
dataset_name = 'mnist'
root_dir = os.getcwd()
dataset_dir = join(root_dir, '../data', dataset_name)
print('MNIST directory: ' + dataset_dir)


# augment labels for new MNIST2class dataset -- training data
train_data, train_labels = torch.load(join(dataset_dir, 'processed/training.pt'))
print(np.shape(train_data))
print(np.shape(train_labels))
print(train_labels)

labels_new = []
for label in train_labels:
    if label % 2 == 0:
        labels_new.append(0)
    else:
        labels_new.append(1)
print(labels_new)
train_labels_2class = torch.tensor(labels_new, dtype=int)
print(train_labels_2class)
# save as torch pt file
torch.save([train_data, train_labels_2class], 'training.pt')


# augment labels for new MNIST2class dataset -- test data
test_data, test_labels = torch.load(join(dataset_dir, 'processed/test.pt'))
print(np.shape(test_labels))
print(test_labels)

labels_new = []
for label in test_labels:
    if label % 2 == 0:
        labels_new.append(0)
    else:
        labels_new.append(1)
print(labels_new)
test_labels_2class = torch.tensor(labels_new, dtype=int)
print(test_labels_2class)
# save as torch pt file
torch.save([test_data, test_labels_2class], 'test.pt')


# new dir for new MNIST2class dataset
new_dir = join(root_dir, '../data', 'mnist2class')
# check if directory exists or not yet
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
save_dir = join(new_dir, 'processed')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print('MNIST2class data results directory: ' + save_dir)

# move created files
shutil.move('training.pt', save_dir)
shutil.move('test.pt', save_dir)
