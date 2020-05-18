import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from os.path import join
import numpy as np
import datasets
import train_utils

dataset_name = 'mnist2class'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = os.getcwd()
model_dir = join(root_dir, 'data', dataset_name, 'models')


parser = train_utils.train_args_parser()
parser.add_argument('--model')  ## add parser arg for model selection
args = parser.parse_args()
batch_size, epochs, lr, run_name = train_utils.parse_train_args(args)
model_name = args.model
print(model_name)

## model instantiation
# for file in os.listdir(model_dir):
#     print(file)
#     # Load model
#     model = torch.load(join(models_dir, file))
#

model_ft = torch.load(join(model_dir, model_name))
print(model_ft)

# # print weights to check if change
# l1 = list(model_ft.fc1.parameters())
# print(l1)
# print(list(model_ft.fc2.parameters()))

# new fine-tune layers
model_ft.fc1 = nn.Linear(1600, 128)
model_ft.fc2 = nn.Linear(128, 10)

# # print newly initalized weights
# lf1 = list(model_ft.fc1.parameters())
# print(lf1)
# print(list(model_ft.fc2.parameters()))
model_ft = model_ft.to(device)

## from train
dataset_dir = join(root_dir, 'data', 'mnist')

dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)
criterion = F.cross_entropy

# loaders
train_loader = dataset.get_train_loader(batch_size=batch_size)
test_loader = dataset.get_test_loader(batch_size=batch_size)


# Training
optimizer = optim.Adam(model_ft.parameters(), lr=lr)  ## Adam instead of SGD
train_acc, train_loss, test_acc, test_loss = train_utils.train(model=model_ft, train_loader=train_loader,
                                                               test_loader=test_loader, optimizer=optimizer,
                                                               device=device, criterion=criterion, epochs=epochs,
                                                               output_dir=dataset_dir, run_name = run_name)
print('Done.')
