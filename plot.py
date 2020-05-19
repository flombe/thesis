import torch
import datasets
import os
from os.path import join
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib


dataset_name = 'mnist'

root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)
print('Dataset directory -- ' + dataset_dir)
model_dir = join(dataset_dir, 'models')


checkpoints = ['0batch0','0batch1','0batch2','0batch3','0batch4','0batch5','0batch6','0batch7','0batch8','0batch9',
               '0batch100','0batch200','0batch300','0batch400','0batch500','0batch600','0batch700','0batch800','0batch900',
               '1','2','3','4','5','7',
               '10','20','30','40','50',
               '100','150','200']
x = np.array([1,2,3,4,5,6,7,8,9,10,100,200,300,400,500,600,700,800,900], dtype=int)
bs = x/937.5
ep = np.array([1,2,3,4,5,7,10,20,30,40,50,100,150,200], dtype=int)
total = np.append(bs, ep)


# Plot the training curves of validation accuracy vs. number of training epochs
#  for the transfer learned models

stats_0batch0 = join(dataset_dir, 'model_ft_mnist2_mnist_0batch0_train_stats.json')
stats_0batch500 = join(dataset_dir, 'model_ft_mnist2_mnist_0batch500_train_stats.json')
stats_1 = join(dataset_dir, 'model_ft_mnist2_mnist_1_train_stats.json')
stats_10 = join(dataset_dir, 'model_ft_mnist2_mnist_10_train_stats.json')
stats_100 = join(dataset_dir, 'model_ft_mnist2_mnist_100_train_stats.json')

stats = [stats_0batch0, stats_0batch500, stats_1, stats_10, stats_100]
label = ['_0batch0', '_0batch500', '_1', '_10', '_100']
color = ['r','b','g','c','m']

fig1, ax1 = plt.subplots()
plt.title("Post Fine-tune Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("post-ft Accuracy")

i = 0
for file in stats:
    with open(file, 'r') as myfile:
        data = myfile.read()
    obj = json.loads(data)
    test_acc = obj['test_acc']
    print('Accuracy: ',test_acc)

    ax1.plot(total, test_acc, color=color[i], label=label[i])
    i=i+1

plt.ylim((0,100))
plt.xscale("log")
plt.xticks(total, rotation=80)
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.legend()
plt.show()




## Detail graph

fig2, ax2 = plt.subplots()
plt.title("Post Fine-tune Accuracy vs. Nr. Train Epochs >1")
plt.xlabel("Training Epochs")
plt.ylabel("post-ft Accuracy")
i=0
for file in stats:
    with open(file, 'r') as myfile:
        data = myfile.read()
    obj = json.loads(data)
    test_acc = obj['test_acc']
    print('Accuracy: ',test_acc)

    #plt.plot(range(1, 15), test_acc[19:33], label=label[i])
    ax2.plot([1, 2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 100, 150, 200], test_acc[19:33],color=color[i], label=label[i])
    i= i+1


plt.ylim((97.5,99.5))
plt.xscale("log")
plt.xticks([1,2,3,4,5,7,10,20,30,40,50,100,150,200])
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.legend(frameon=False, loc='lower center', ncol=2)
plt.show()
