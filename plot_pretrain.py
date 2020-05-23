import os
from os.path import join
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib

# plot pretraining on MNIST2class with stat test of 10 seeds
dataset_name = 'mnist2class'

root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)
print('Dataset directory -- ' + dataset_dir)


# for log scale use xsteps = [0.001, 0.01, 0.1, 1, 10, 100] eql. to 1, 10, 100 batches, 1,10,100 epochs
# for little finer plot add 3s (roughly halfway on log-scale)
checkpts = ['0batch1','0batch3','0batch10','0batch30','0batch100','0batch300','1','3','10','30','100','200']
xticks = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100]

x = np.array([1,3,10,30,100,300], dtype=int)
bs = x/937.5
ep = np.array([1,3,10,30,100], dtype=int)
total = np.append(bs, ep)

# aggregate Acc's from 10 seed runs
run_name = 'pre_mnist2_'

acc_dict = {}
accs = np.zeros((10, 11))

for seed in range(1,11):
    model_dir = join(dataset_dir, 'models_'+str(seed))
    batch_stats = join(model_dir, run_name + '_batch_train_stats.json')
    train_stats = join(model_dir, run_name + '_train_stats.json')

    # concat batch and epoch stats for Acc
    test_acc = []
    for file in [batch_stats, train_stats]:
        with open(file, 'r') as myfile:
            data = myfile.read()
        obj = json.loads(data)
        test_acc += obj['test_acc']
    print(test_acc)

    accs[seed-1] = test_acc

print(accs.shape)


# Plot accuracy vs. number of training epochs
fig1, ax1 = plt.subplots()
plt.title("Pre-train Accuracy vs. Nr of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Pre-train Accuracy")

for i in range(accs.shape[0]):
    ax1.plot(total, accs[i],'x', label=checkpts[i])

median = []
std = []
for i in range(accs.shape[1]):
    median.append(np.median(accs[:,i]))
    std.append(np.std(accs[:,i]))
ax1.errorbar(total, median, yerr=std, linewidth=2, label='median')

plt.ylim((40,100))
plt.xscale("log")
plt.xticks(xticks, rotation=80)
f = lambda x,pos: str(x).rstrip('0').rstrip('.')
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(f))
ax1.xaxis.set_tick_params(which='minor', bottom=False)
plt.legend(loc=4)
plt.show()