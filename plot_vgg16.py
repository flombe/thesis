import os
from os.path import join
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from natsort import natsorted

###########
pre_dataset = 'imagenet'
ft_dataset = 'custom3D'

plot_acc = True
###########

root_dir = os.getcwd()
models_dir = join(root_dir, 'models')

# ticks for plot - batches and epochs with bs=12 and 1200 samples
checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '1', '3', '10', '30', '100']
xticks = [0.0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
total = np.array(xticks)

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Paired.colors)  # set color scheme


# Plot Acc on VGG16 custom3D for different ft or pre
if plot_acc:

    pre_dataset = ['random_init', 'imagenet', 'custom3D']

    fig1, ax1 = plt.subplots(figsize=(6, 7), dpi=150)
    plt.title(f"Accuracies of VGG-16 models on {ft_dataset}")
    plt.xlabel("Fine-Tuning/training Epochs (batch1 to epoch100)")
    plt.ylabel("Test Accuracy")

    for dataset in pre_dataset:

        if dataset == ft_dataset:
            load_dir = join(models_dir, dataset, 'models_1')
            label = f'pre_{dataset}'
            # load Acc from json file
            with open(join(load_dir, label + '_train_stats.json'), 'r') as myfile:
                data = myfile.read()
            test_acc = json.loads(data)['pre_test_acc']
        else:
            load_dir = join(models_dir, dataset, 'ft_' + ft_dataset)
            label = f"ft_{dataset}_{ft_dataset}"
            # load Acc from df
            df = pd.read_pickle(join(load_dir, "df_" + label))
            test_acc = df['ft_test_acc']

        ax1.plot(total, test_acc, label=str(label))

    # additional
    for add_case in ['_lastlayer', '_3conv', '_onlyfc']:
        load_dir = join(models_dir, 'imagenet', 'ft_' + ft_dataset + add_case)
        label = f"ft_imagenet_{ft_dataset}{add_case}"
        # load Acc from df
        df = pd.read_pickle(join(load_dir, "df_" + label))
        test_acc = df['ft_test_acc']

        ax1.plot(total, test_acc, label=str(label))


    plt.ylim((0, 100))
    plt.xscale("log")

    ax1.axis([0, 100, 0, 100])
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()
