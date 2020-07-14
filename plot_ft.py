import os
from os.path import join
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# plot fine-tune on dataset_name on pre-trained models of 10 seeds
pre_dataset = 'mnist'
ft_dataset = 'fashionmnist'

root_dir = os.getcwd()
models_dir = join(root_dir, 'models', pre_dataset, 'ft_' + ft_dataset)

# run_name = join(f'ft_{pre_dataset}_{ft_dataset}_')  # 'ft_mnist2_mnist_'
run_name = join(f'ft_{pre_dataset}_fashion_')

# for log scale use xsteps = [0.001, 0.01, 0.1, 1, 10, 100] eql. to 1, 10, 100 batches, 1,10,100 epochs
# for little finer plot add 3s (roughly halfway on log-scale)
checkpts = ['0_1', '0_3', '0_10', '0_30', '0_100', '0_300', '1', '3', '10', '30', '100']
xticks = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100]

x = np.array([1,3,10,30,100,300], dtype=int)
bs = x/937.5
ep = np.array([1,3,10,30,100], dtype=int)
total = np.append(bs, ep)


# aggregate Acc's from 10 seed runs of 11 different models in dict
mydict = dict()
for check in checkpts:
    accs = np.zeros((10, 11))
    for seed in range(3, 4):  # range(1, 11)
        seed_dir = join(models_dir, 'models_' + str(seed))
        train_stats = join(seed_dir, run_name + check + '_train_stats.json')

        test_acc = []
        with open(train_stats, 'r') as myfile:
            data = myfile.read()
        obj = json.loads(data)
        test_acc += obj['ft_test_acc']

        accs[seed-1] = test_acc
    mydict.update({check: accs})
print(mydict)


### load df instead
# df = pd.read_pickle(join(models_dir, "df_" + run_name + ".pkl"))





# Plot post-ft accuracy vs. number of training epochs
fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=150)
plt.title("Post-Finetune Accuracy")  # (mean of 10 seeds)")
plt.xlabel("Fine-Tuning Epochs (batch1 to epoch100)")
plt.ylabel("Post-Ft Accuracy")

for check in mydict.keys():
    accs = mydict[check][0] # fist line, since only models_1
    ax1.plot(total, accs, label=(str(check)))

plt.ylim((0, 100))
ax1.minorticks_on()
plt.xscale("log")
plt.xticks(xticks, rotation=80)
f = lambda x,pos: str(x).rstrip('0').rstrip('.')
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(f))
ax1.xaxis.set_tick_params(which='minor', bottom=False)
plt.legend(loc=4)
plt.tight_layout()
plt.show()




# Plot post-ft accuracy vs. number of training epochs
fig3, ax3 = plt.subplots(figsize=(7, 6), dpi=150)
plt.title("Detail view: Post-Finetune Accuracy")
plt.xlabel("Fine-Tuning Epochs")
plt.ylabel("Post-Ft Accuracy")

for check in mydict.keys():
    accs = mydict[check][0]
    ax3.plot(total[6:], accs[6:], label=(str(check)))

plt.ylim((79, 93))
ax3.minorticks_on()
plt.xscale("log")
plt.xticks(xticks[6:])
ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax3.xaxis.set_tick_params(which='minor', bottom=False)
plt.legend(loc=4)
plt.tight_layout()
plt.show()







# Plot post-ft accuracy details for 0batch0 and 100 vs. number of training epochs
fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=150)
plt.title("Shortest/longest Pre-trained: Post-Ft Accuracy")
plt.xlabel("Fine-Tuning Epochs (batch1 to epoch100)")
plt.ylabel("Post-Ft Accuracy")

for check in ['0_1', '100']:
    accs = mydict[check][0]
    ax2.plot(total, accs, label=(str(check)))

plt.ylim((0,100))
ax2.minorticks_on()
plt.xscale("log")
plt.xticks(xticks, rotation=80)
f = lambda x,pos: str(x).rstrip('0').rstrip('.')
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(f))
ax2.xaxis.set_tick_params(which='minor', bottom=False)
plt.legend(loc=4)
plt.tight_layout()
plt.show()



## New plot - show pre-train time on xaxis

# Plot post-ft accuracy vs. number of pre-training epochs
from matplotlib import gridspec
fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=150)
plt.title("Post-Finetune Accuracy vs. Pre-train duration")
plt.xlabel("Pre-Training Epochs (batch1 to epoch100)")
plt.ylabel("Post-Ft Accuracy")

lines = np.zeros((11, 11))
j = 0
for check in mydict.keys():
    accs = mydict[check]
    mean = []
    std = []
    for i in range(accs.shape[1]):
        mean.append(np.sum(accs[:,i]))
    lines[j] = mean
    j+=1

for i in range(11):
    ax1.plot(total, lines[:,i], label=(checkpts[i]))

plt.ylim((0,100))
ax1.minorticks_on()
plt.xscale("log")
plt.xticks(xticks, rotation=80)
f = lambda x,pos: str(x).rstrip('0').rstrip('.')
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(f))
ax1.xaxis.set_tick_params(which='minor', bottom=False)
plt.legend(loc=4, title='fine-tuned for')
plt.tight_layout()
plt.show()


