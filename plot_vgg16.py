import os
from os.path import join
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from natsort import natsorted


###########
pre_dataset = 'segnet'  # 'imagenet'
ft_dataset = 'custom3D'

plot_acc = False

# general plots over all datasets
plot_acc_all = False
plot_acc_all_delta = False

plot_ss_id_all = True
###########


root_dir = os.getcwd()
models_dir = join(root_dir, 'models', 'vgg16')

# ticks for plot - batches and epochs with bs=12 and 1200 samples
checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '1', '3', '10', '30', '100']
xticks = [0.0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
total = np.array(xticks)

# Plot Acc on VGG16 custom3D for different ft or pre cases
if plot_acc:
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Paired.colors)  # set color scheme
    pre_datasets = ['random_init', pre_dataset, 'custom3D']

    fig1, ax1 = plt.subplots(figsize=(6, 7), dpi=150)
    plt.title(f"Accuracies of VGG-16 models on {ft_dataset}")
    plt.xlabel("Fine-Tuning/training Epochs (batch1 to epoch100)")
    plt.ylabel("Test Accuracy")

    for dataset in pre_datasets:

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
        print(test_acc)
        ax1.plot(total, test_acc, label=str(label))

    # additional
    for add_case in ['_lastlayer', '_3conv', '_onlyfc']:
        load_dir = join(models_dir, pre_dataset, 'ft_' + ft_dataset + add_case)
        label = f"ft_{pre_dataset}_{ft_dataset}{add_case}"
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

# Plot ft Acc on VGG16 custom3D for different pre_datasets
if plot_acc_all:
    pre_datasets = ['imagenet', 'places365', 'cars', 'vggface', 'segnet', 'cifar10', 'random_init', 'custom3D']

    fig1, ax1 = plt.subplots(figsize=(6, 7), dpi=150)
    plt.title(f"Accuracies of VGG-16 models on {ft_dataset}")
    plt.xlabel("Fine-Tuning/training Epochs (batch1 to epoch100)")
    plt.ylabel("Test Accuracy")

    for dataset in pre_datasets:

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
        print(test_acc)
        ax1.plot(total, test_acc, label=str(label))

    plt.ylim((0, 100))
    plt.xscale("log")

    ax1.axis([0, 100, 0, 100])
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()

# Plot Acc Delta: post-ft vs. base-case  +std
if plot_acc_all_delta:
    pre_datasets = ['imagenet', 'places365', 'cars', 'vggface', 'segnet', 'cifar10', 'random_init']
    baseline = 'custom3D'

    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    plt.title(f"Accuracy Delta to Baseline for VGG-16 models on {ft_dataset}")
    plt.xlabel("Fine-Tuning/training Epochs (batch1 to epoch100)")
    plt.ylabel("Test Accuracy Delta")

    #####  - run multiple pre on custom3D --> take mean of Accs. and also plot std

    # baseline = pre_custom3D - get mean
    load_dir = join(models_dir, baseline, 'models_1')
    label = f'pre_{baseline}'
    # load Acc from json file
    with open(join(load_dir, label + '_train_stats.json'), 'r') as myfile:
        data = myfile.read()
    base_acc = json.loads(data)['pre_test_acc']

    #####

    for dataset in pre_datasets:
        load_dir = join(models_dir, dataset, 'ft_' + ft_dataset)
        label = f"ft_{dataset}_{ft_dataset}"
        # load Acc from df
        df = pd.read_pickle(join(load_dir, "df_" + label))
        test_acc = df['ft_test_acc']
        print(test_acc)
        ax.plot(total, test_acc - base_acc, label=str(label))

    base = ax.axhline(linewidth=0.5, color='k')

    # std
    # ax.fill_between(total[1:], 2 * target_stds, -2 * target_stds, color=base.get_color(), alpha=0.05, label='std_base')

    plt.ylim((-25, 90))
    plt.xscale("log")
    plt.xlim((0, 100))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()


if plot_ss_id_all:
    pre_datasets = ['imagenet', 'places365', 'cars', 'vggface', 'segnet', 'cifar10']

    xticks = ['in', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc1', 'fc2', 'out']
    plt.style.use('seaborn')
    #plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20c.colors)  # set color scheme

    fig, axs = plt.subplots(2, sharex=True, figsize=(7, 9), dpi=150)
    axs[0].set_title(f'ID and SumSqr over VGG model layers \n [extract: custom3D]', weight='semibold')

    for dataset in pre_datasets:
        df_path = join(models_dir, dataset, 'df_pre_' + dataset + '+metrics')
        df = pd.read_pickle(df_path)

        # load from df
        id = df['ID_custom3D'][0]
        ss = df['SS_custom3D'][0]

        # since on cifar10 no extract possible on fc layers and for segnet no fc pre layers
        if dataset == 'segnet':
            id = id[:6]
            ss = ss[:6]
        if dataset in ['segnet', 'cifar10']: x_range = range(6)
        else: x_range = range(len(xticks))

        print(dataset, x_range, id)
        axs[0].plot(x_range, id, '.-')
        axs[1].plot(x_range, ss, '.-', label=df['model_name'][0])

    axs[0].set_ylabel("Intrinsic Dimension", weight='semibold')
    plt.ylabel("SSW/TSS", weight='semibold')

    plt.xlabel("Layers", weight='semibold')
    plt.xticks(range(len(xticks)), labels=xticks)

    plt.legend(loc="lower left", frameon=True, fancybox=True, facecolor='white', title='diff. pre datasets')
    plt.show()
