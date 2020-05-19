import torch
import datasets
import os
from os.path import join
import numpy as np
from train_utils import evaluate
import json


dataset_name = 'mnist'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)
print('Dataset directory -- ' + dataset_dir)
model_dir = join(dataset_dir, 'models')

# model instantiation
# model = torch.load(join(model_dir, 'model_6.pt'))
# model.eval()
# model.to(device)

# dataset = datasets.MNIST2class(dataset_dir=dataset_dir, device=device)
# train_data, train_labels = torch.load(join(dataset_dir, 'MNIST/processed/training.pt'))
# print(np.shape(train_data))
# print(train_labels)
#
# train_loader = dataset.get_train_loader(batch_size=100)
# print(train_loader)
# print(len(train_loader))
#
# for i, (images, labels) in enumerate(train_loader):
#     print(type(images))
#     print(labels)

batch_size = 64
dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)

train_loader = dataset.get_train_loader(batch_size=batch_size)
test_loader = dataset.get_test_loader(batch_size=batch_size)

# model instantiation
for file in os.listdir(model_dir):
    print(file)
    # Load model
    model = torch.load(join(model_dir, file))

    run_name = str(file)

    # training and information collection
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []


    model.eval()
    loss, acc = evaluate(model, train_loader, device)
    train_loss.append(loss.item())
    train_acc.append(acc)

    loss, acc = evaluate(model, test_loader, device)
    test_loss.append(loss.item())
    test_acc.append(acc)

    train_stats = {
        'model_cls': model.__class__.__name__,
        'run_name': run_name,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'test_acc': test_acc,
        'test_loss': test_loss
    }
    stats_file_path = join(model_dir, run_name+'_train_stats.json')  ## otherwise override json files for same model name but different runs
    with open(stats_file_path, 'w+') as f:
        json.dump(train_stats, f)

    print(run_name + ' done')







### Experiment 1:  Pre-training to TL Performance Relation

## create augmented MNIST2class dataset
# --> MNIST2class.py

## pre-train on MNIST2class and safe checkpoints
# Train MNIST models different amount of batches/epochs
# epoch 0 batch 1-10 and mod50
# epoch 1, 3, 5, 7, 10, 20, 30, 40, 50, 100, 150, 200
# safe model
# --> pretrain_mnist2class.sh

## fine-tune models on MNIST
# Transfer Learn and fine-tune different amount of epochs
# finetune.py
# reinitialize fc1 and fc2 layers from loaded models
# --> ft finetune_mnist.sh

## Plot and compare
# plot TL fine-tuned model Acc. on test over epochs (multiple lines are multiple pre-trained models)
