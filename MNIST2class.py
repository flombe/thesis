import torch
import datasets
import os
from os.path import join
import numpy as np


dataset_name = 'mnist'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)
print('Dataset directory -- ' + dataset_dir)
# model_dir = join(dataset_dir, 'models')

# model instantiation
# model = torch.load(join(model_dir, 'model_6.pt'))
# model.eval()
# model.to(device)

dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)
train_data, train_labels = torch.load(join(dataset_dir, 'processed/training.pt'))
print(train_labels)
print(np.shape(train_data))
print(np.shape(train_labels))

# augment labels for new MNIST2class dataset -- training data
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
torch.save([train_data, train_labels_2class], 'train2class.pt')


# augment labels for new MNIST2class dataset -- test data
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
torch.save([train_data, train_labels_2class], 'train2class.pt')




#print("Number of training examples :", X_train.shape[0], "and each image is of shape (%d, %d)"%(X_train.shape[1], X_train.shape[2]))
#print("Number of test examples :", X_test.shape[0], "and each image is of shape (%d, %d)"%(X_test.shape[1], X_test.shape[2]))





### Experiment 1:  Pre-training to TL Performance Relation

# Train MNIST models different amount of epochs

# train less than 1 epoch?
# epoch 1, 3, 5, 10, 20, 50, 100, 200

# safe model

# Transfer Learn and fine-tune different amount of epochs


# Compare and plot TL fine-tuned model Acc. on test over epochs (multiple lines are multiple pre-trained models)
