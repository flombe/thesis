import os
from os.path import join
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib

# plot pretraining on MNIST2class with stat test of 10 seeds
dataset_name = 'mnist2class'
root_dir = os.getcwd()
dataset_dir = join(root_dir, 'models', dataset_name)
run_name = 'pre_mnist2'

xticks = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
x = np.array([1, 3, 10, 30, 100, 300], dtype=int)
bs = x/937.5
ep = np.array([1, 3, 10, 30, 100], dtype=int)
total = np.append(bs, ep)

# load and aggregate Acc's from 10 seed runs
accs = np.zeros((10, 11))

for seed in range(1, 11):
    model_dir = join(dataset_dir, 'models_' + str(seed))
    train_stats = join(model_dir, run_name + '_train_stats.json')

    with open(train_stats, 'r') as myfile:
        data = myfile.read()
    obj = json.loads(data)
    accs[seed-1] = obj['pre_test_acc']

# Plot accuracy vs. number of training epochs
fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=150)
plt.title("Pre-train Accuracy vs. Nr of Training Epochs")
plt.xlabel("Training Epochs (batch1 to epoch100)")
plt.ylabel("Pre-train Accuracy")

for i in range(accs.shape[0]):
    ax1.plot(total, accs[i], 'x', label=('seed '+str(i)))  #label=checkpts[i]

mean = []
std = []
for i in range(accs.shape[1]):
    mean.append(np.mean(accs[:, i]))
    std.append(np.std(accs[:, i]))
p95 = 2*np.array(std)  # [mean-2std, mean+2std] approx 95% percentile

ax1.errorbar(total, mean, yerr=p95, color='k', linewidth=1,
             ecolor='gray', elinewidth=1, capsize=4, label='mean')

plt.ylim((40, 100))
plt.xscale("log")
plt.xticks(xticks, rotation=80)
f = lambda x, pos: str(x).rstrip('0').rstrip('.')
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(f))
ax1.xaxis.set_tick_params(which='minor', bottom=False)
plt.legend(loc=4)
plt.show()
print('Plot Pretraining')
