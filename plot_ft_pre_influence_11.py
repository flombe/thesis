import os
from os.path import join
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from natsort import natsorted

# plot fine-tune on dataset_name on pre-trained models of 10 seeds
pre_dataset = 'fashionmnist'
ft_dataset = 'mnist'

root_dir = os.getcwd()
models_dir = join(root_dir, 'models', pre_dataset, 'ft_' + ft_dataset)

run_name = join(f'ft_{pre_dataset}_{ft_dataset}_')  # 'ft_mnist2_mnist_'
if ft_dataset == 'fashionmnist': run_name = join(f'ft_{pre_dataset}_fashion_')  # naming only fashion


checkpts = ['0_1', '0_3', '0_10', '0_30', '0_100', '0_300', '1', '3', '10', '30', '100']
xticks = [0.0, 0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100]

x = np.array([0, 1,3,10,30,100,300], dtype=int)
bs = x/937.5
ep = np.array([1,3,10,30,100], dtype=int)
total = np.append(bs, ep)

# aggregate Acc's from 10 seed runs of 11 different models in dict
mydict = dict()
for check in checkpts:
    accs = np.zeros((10, 12))
    for seed in range(1, 11):
        model_dir = join(models_dir, 'models_' + str(seed))
        train_stats = join(model_dir, run_name + check + '_train_stats.json')

        # concat batch and epoch stats for Acc
        test_acc = []
        with open(train_stats, 'r') as myfile:
            data = myfile.read()
        obj = json.loads(data)
        test_acc += obj['ft_test_acc']

        accs[seed-1] = test_acc
    mydict.update({check: accs})
#print(mydict)



### load df instead
target_model_dir = join(root_dir, 'models', ft_dataset)
if ft_dataset == 'fashionmnist':
    df = pd.read_pickle(join(target_model_dir, "df_pre_fashion"))
else:
    df = pd.read_pickle(join(target_model_dir, "df_pre_" + ft_dataset))

# get mean 'pre_test_acc' for every model_name of target task
target_means = df.groupby('model_name')['pre_test_acc'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
target_stds = df.groupby('model_name')['pre_test_acc'].apply(lambda g: np.std(g.values.tolist(), axis=0))
target_means = target_means.reindex(index=natsorted(target_means.index))
target_stds = target_stds.reindex(index=natsorted(target_stds.index))



# Plot post-ft accuracy vs. number of training epochs
fig1, ax1 = plt.subplots(figsize=(6, 7), dpi=150)
plt.title(f"pre: {pre_dataset}, ft: {ft_dataset} \n Post-ft Accuracies vs. training on target (mean of 10 seeds)")
plt.xlabel("Fine-Tuning/training Epochs (batch1 to epoch100)")
plt.ylabel("Post-Ft/training Test Accuracy")

for check in mydict.keys():
    accs = mydict[check]
    # for i in range(accs.shape[0]):
    #     ax1.plot(total, accs[i], 'x') #, label=(str(check) +' ft_ '+str(i)))

    mean = []
    std = []
    for i in range(accs.shape[1]):
        mean.append(np.mean(accs[:,i]))
        std.append(np.std(accs[:,i]))
    p95 = 2*np.array(std)  ## [mean-2std, mean+2std] approx 95% percentile
    ax1.plot(total, mean, label=(str(check)))  # +' mean'

# plot base case: training target normaly - use mean of 10 seeds and plot std shade
base = ax1.plot(total[1:], target_means, '-.', label='base_case', c='k')
ax1.fill_between(total[1:], target_means - 2 * target_stds, target_means + 2 * target_stds, color=base[0].get_color(),
                 alpha=0.1, label='std_base')

plt.ylim((0,100))
ax1.minorticks_on()
plt.xscale("log")
# plt.xticks(xticks, rotation=80)
# f = lambda x,pos: str(x).rstrip('0').rstrip('.')
# ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(f))
# ax1.xaxis.set_tick_params(which='minor', bottom=False)
plt.legend(loc=4)
plt.tight_layout()
plt.show()



# Plot Acc Delta post-ft vs. base-case

fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=150)
plt.title(f"pre: {pre_dataset}, ft: {ft_dataset} \n Delta Post-ft Acc. - base-case Acc.")
plt.xlabel("Fine-Tuning/training Epochs (batch1 to epoch100)")
plt.ylabel("Post-Ft/training Test Accuracy")

for check in mydict.keys():
    accs = mydict[check]

    mean = []
    for i in range(accs.shape[1]):
        mean.append(np.mean(accs[:,i]))
    ax2.plot(total[1:], mean[1:]-target_means, label=(str(check)))

ax2.axhline(linewidth=0.5, color='lightgrey')

plt.ylim((-5, 25))
ax2.minorticks_on()
plt.xscale("log")
# plt.xticks(xticks, rotation=80)
# f = lambda x,pos: str(x).rstrip('0').rstrip('.')
# ax2.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(f))
# ax2.xaxis.set_tick_params(which='minor', bottom=False)
plt.legend(loc=1)
plt.tight_layout()
plt.show()

