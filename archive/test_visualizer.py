import numpy as np
import torch
from tqdm import tqdm
import os
from os.path import join
import pandas as pd
import datasets
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)


trained_dataset = 'mnist_noise_struct'
target_dataset = 'mnist'

root_dir = os.getcwd()
models_dir = join(root_dir, '../models', trained_dataset)

# load df
df_path = join(models_dir, 'df_pre_mnist_noise_struct')
df = pd.read_pickle(df_path)


# load dataset
dataset_dir = join(root_dir, '../data', trained_dataset)
dataset = datasets.MNIST_noise_struct(dataset_dir=dataset_dir, device=device)
train_loader = dataset.get_train_loader(batch_size=1)

for i in range(10):
    image, label = next(iter(train_loader))
    print(label)
    print(image.mean())
    plt.figure(figsize=(3, 3))
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(label)
    plt.show()
