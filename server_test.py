import torch
import datasets
import os
from os.path import join
import numpy as np


dataset_name = 'mnist2class'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)
print('Dataset directory -- ' + dataset_dir)
# model_dir = join(dataset_dir, 'models')

# model instantiation
# model = torch.load(join(model_dir, 'model_6.pt'))
# model.eval()
# model.to(device)

dataset = datasets.MNIST2class(dataset_dir=dataset_dir, device=device)
train_data, train_labels = torch.load(join(dataset_dir, 'processed/training.pt'))
print(np.shape(train_data))
print(train_labels)





### Experiment 1:  Pre-training to TL Performance Relation

## create augmented MNIST2class dataset
# --> MNIST2class.py

### train less than 1 epoch !! ###

# Train MNIST models different amount of epochs
# epoch 1, 3, 5, 7, 10, 20, 30, 40, 50, 100, 150, 200
# safe model
# --> pretrain_mnist2class.sh


# Transfer Learn and fine-tune different amount of epochs


# Compare and plot TL fine-tuned model Acc. on test over epochs (multiple lines are multiple pre-trained models)
