import os
from os.path import join
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import pandas

# plot fine-tune on dataset_name on pre-trained models of 10 seeds
pre_dataset = 'mnist_split2'
ft_dataset = 'mnist'

root_dir = os.getcwd()
models_dir = join(root_dir, 'models', pre_dataset, 'ft_' + ft_dataset)

run_name = join(f'ft_{pre_dataset}_{ft_dataset}_')  # 'ft_mnist2_mnist_'
# run_name = join(f'ft_{pre_dataset}_fashion_')

# for log scale use xsteps = [0.001, 0.01, 0.1, 1, 10, 100] eql. to 1, 10, 100 batches, 1,10,100 epochs
# for little finer plot add 3s (roughly halfway on log-scale)
checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '0_100', '0_300', '1', '3', '10', '30', '100']
xticks = [0.0, 0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100]

x = np.array([0, 1,3,10,30,100,300], dtype=int)
bs = x/937.5
ep = np.array([1,3,10,30,100], dtype=int)
total = np.append(bs, ep)

# aggregate Acc's from 10 seed runs of 11 different models in dict
mydict = dict()
for check in checkpts:
    accs = np.zeros((10, 12))
    for seed in range(1,11):
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
# df = pd.read_pickle(join(models_dir, "df_" + run_name + ".pkl"))




# Plot post-ft accuracy vs. number of training epochs
fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=150)
ax1.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)
plt.title("Post-Finetune Accuracy (mean of 10 seeds)")
plt.xlabel("Fine-Tuning Epochs (batch1 to epoch100)")
plt.ylabel("Post-Ft Accuracy")

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

    ax1.plot(total, mean, label=(str(check))) # +' mean'

plt.ylim((0,100))
ax1.minorticks_on()
plt.xscale("log")
# plt.xticks(xticks, rotation=80)
ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
ax1.xaxis.set_tick_params(which='minor', bottom=False)
plt.legend(loc=4)
plt.tight_layout()
plt.show()



# Plot post-ft accuracy vs. number of training epochs
fig3, ax3 = plt.subplots(figsize=(7, 6), dpi=150)
ax3.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)
plt.title("Detail view: Post-Finetune Accuracy (mean)")
plt.xlabel("Fine-Tuning Epochs")
plt.ylabel("Post-Ft Accuracy")

for check in mydict.keys():
    accs = mydict[check]
    mean = []
    std = []
    for i in range(accs.shape[1]):
        mean.append(np.mean(accs[:,i]))
        std.append(np.std(accs[:,i]))
    p95 = 2*np.array(std)

    if check in ['0_1', '100']:
        #for i in range(accs.shape[0]):
        #    ax1.plot(total[6:], accs[i][6:], 'x')

        ax3.errorbar(total[6:], mean[6:], yerr=p95[6:],
                     # ecolor='gray',
                     elinewidth=1, capsize=4,
                     label=(str(check)))

    else: ax3.plot(total[6:], mean[6:], label=(str(check)))

# plt.ylim((78, 93))
ax3.minorticks_on()
plt.xscale("log")
# plt.xticks(xticks[6:])
ax3.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
ax3.xaxis.set_tick_params(which='minor', bottom=False)
plt.legend(loc=4)
plt.tight_layout()
plt.show()







# Plot post-ft accuracy details for 0batch0 and 100 vs. number of training epochs
fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=150)
ax2.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)
plt.title("Shortest/longest Pre-trained: Post-Ft Accuracy")
plt.xlabel("Fine-Tuning Epochs (batch1 to epoch100)")
plt.ylabel("Post-Ft Accuracy")

for check in ['0_1', '100']:
    accs = mydict[check]
    for i in range(accs.shape[0]):
        ax2.plot(total, accs[i], 'x') #, label=(str(check) +' ft_ '+str(i)))

    mean = []
    std = []
    for i in range(accs.shape[1]):
        mean.append(np.mean(accs[:,i]))
        std.append(np.std(accs[:,i]))
    p95 = 2*np.array(std)  ## [mean-2std, mean+2std] approx 95% percentile

    ax2.errorbar(total, mean, yerr=p95,
                 #ecolor='gray',
                 elinewidth=1, capsize=4,
                 label=(str(check))) # +' mean'

plt.ylim((0,100))
ax2.minorticks_on()
plt.xscale("log")
# plt.xticks(xticks, rotation=80)
ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
ax2.xaxis.set_tick_params(which='minor', bottom=False)
plt.legend(loc=4)
plt.tight_layout()
plt.show()



## New plot - show pre-train time on xaxis

# Plot post-ft accuracy vs. number of pre-training epochs
fig1, ax1 = plt.subplots(figsize=(7, 6), dpi=150)
ax1.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)
plt.title("Post-Finetune Accuracy (means) vs. Pre-train duration")
plt.xlabel("Pre-Training Epochs (batch1 to epoch100)")
plt.ylabel("Post-Ft Accuracy")
plt.ylabel("Detail Post-Ft Accuracy")

lines = np.zeros((12, 12))
j = 0
for check in mydict.keys():
    accs = mydict[check]
    mean = []
    std = []
    for i in range(accs.shape[1]):
        mean.append(np.mean(accs[:,i]))
        std.append(np.std(accs[:,i]))
    p95 = 2*np.array(std)  ## [mean-2std, mean+2std] approx 95% percentile

    lines[j] = mean
    j+=1

for i in range(11):
    ax1.plot(total, lines[:,i], label=(checkpts[i]))

plt.ylim((0,100))
ax1.minorticks_on()
plt.xscale("log")
# plt.xticks(xticks, rotation=80)
ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
ax1.xaxis.set_tick_params(which='minor', bottom=False)
plt.legend(loc=4, title='fine-tuned for')
plt.tight_layout()
plt.show()


