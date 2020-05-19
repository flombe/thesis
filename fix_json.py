import torch
import datasets
import os
from os.path import join
import numpy as np
import json
from pathlib import Path

dataset_name = 'mnist'

root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)
model_dir = join(dataset_dir, 'models')


paths = sorted(Path(model_dir).iterdir(), key=os.path.getmtime) ## sort by system time created
print(paths)
print(paths[132])
print(paths[164])
print(paths[165])

run_name = "model_ft_mnist2_mnist_100_"
i = 0
## combine train_stats.json

train_loss = []
train_acc = []
test_loss = []
test_acc = []

#for filename in os.listdir(model_dir):
#    if filename.startswith(str(run_name)) and filename.endswith(".json"):
for path in paths[132:165]: # +1
         i = i+1
         file = join(str(path) + '_train_stats.json')
         print(file)

         with open(file, 'r') as myfile:
             data = myfile.read()
         # parse file
         obj = json.loads(data)

         #print(obj['train_acc'])
         train_acc = train_acc + obj['train_acc']
         train_loss = train_loss + obj['train_loss']
         test_acc = test_acc + obj['test_acc']
         test_loss = test_loss + obj['test_loss']

print(i)

train_stats = {
        'model_cls': "Net",
        'run_name': run_name,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'test_acc': test_acc,
        'test_loss': test_loss
}
stats_file_path = join(dataset_dir, run_name + 'train_stats.json')
with open(stats_file_path, 'w+') as f:
     json.dump(train_stats, f)
print('Done')


