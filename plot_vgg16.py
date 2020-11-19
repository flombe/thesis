import os
from os.path import join
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from natsort import natsorted
from rsa import get_rdm_metric_vgg


root_dir = os.getcwd()
models_dir = join(root_dir, 'models', 'vgg16')

# Plot Acc on VGG16 custom3D for different ft or pre cases
def plot_acc():
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
def plot_acc_all():
    pre_datasets = ['imagenet', 'places365', 'cars', 'vggface', 'segnet', 'cifar10', 'random_init', ft_dataset]

    fig1, ax1 = plt.subplots(figsize=(6, 7), dpi=150)
    plt.title(f"Accuracies of VGG-16 models on {ft_dataset}")
    plt.xlabel("Fine-Tuning/training Epochs (batch1 to epoch100)")
    plt.ylabel("Test Accuracy")

    for dataset in pre_datasets:

        if dataset in ['random_init', ft_dataset]:
            load_dir = join(models_dir, dataset, 'ft_' + ft_dataset)
            label = f"ft_{dataset}_{ft_dataset}"
            case = 'ft'
            if dataset == ft_dataset:
                load_dir = join(models_dir, dataset)
                label = f'pre_{dataset}'
                case = 'pre'

            # load Acc from df
            df = pd.read_pickle(join(load_dir, "df_" + label))
            # get mean and std 'pre_test_acc' over model_names for 10 (3) seeds
            base_means = df.groupby('model_name')[f'{case}_test_acc'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
            base_stds = df.groupby('model_name')[f'{case}_test_acc'].apply(lambda g: np.std(g.values.tolist(), axis=0))
            base_means = base_means.reindex(index=natsorted(base_means.index))
            base_stds = base_stds.reindex(index=natsorted(base_stds.index))

            print(dataset, base_stds)
            # print(base_means, base_stds)
            base = ax1.plot(total, base_means, label=str(label))
            ax1.fill_between(total, base_means + 2 * np.array(base_stds), base_means - 2 * np.array(base_stds),
                             color=base[0].get_color(), alpha=0.2)
        else:
            load_dir = join(models_dir, dataset, 'ft_' + ft_dataset)
            label = f"ft_{dataset}_{ft_dataset}"
            # load Acc from df
            df = pd.read_pickle(join(load_dir, "df_" + label))
            test_acc = df['ft_test_acc']

            print(label, test_acc)
            ax1.plot(total, test_acc, label=str(label))

    #plt.ylim((40, 100))
    plt.xscale("log")

    ax1.axis([0, 100, 0, 100])
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()

# Plot Acc Delta: post-ft vs. base-case  +std
def plot_acc_all_delta():
    pre_datasets = ['imagenet', 'places365', 'cars', 'vggface', 'segnet', 'cifar10', 'random_init']
    baseline = 'custom3D'

    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    plt.title(f"Accuracy Delta to Baseline for VGG-16 models on {ft_dataset}")
    plt.xlabel("Fine-Tuning/training Epochs (batch1 to epoch100)")
    plt.ylabel("Test Accuracy Delta")

    # baseline = pre_custom3D - get mean
    load_dir = join(models_dir, baseline)
    label = f'pre_{baseline}'
    # load Acc from df
    df = pd.read_pickle(join(load_dir, "df_" + label))
    # get mean and std 'pre_test_acc' over model_names for 10 seeds
    base_means = df.groupby('model_name')['pre_test_acc'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
    base_stds = df.groupby('model_name')['pre_test_acc'].apply(lambda g: np.std(g.values.tolist(), axis=0))
    base_means = base_means.reindex(index=natsorted(base_means.index))
    base_stds = base_stds.reindex(index=natsorted(base_stds.index))

    base = ax.axhline(linewidth=1.5, color='grey')
    ax.fill_between(total, 2 * np.array(base_stds), - 2 * np.array(base_stds), color=base.get_color(), alpha=0.2,
                    label='std_base')

    for dataset in pre_datasets:
        load_dir = join(models_dir, dataset, 'ft_' + ft_dataset)
        label = f"ft_{dataset}_{ft_dataset}"
        # load Acc from df
        df = pd.read_pickle(join(load_dir, "df_" + label))
        test_acc = df['ft_test_acc']
        if dataset == 'random_init':
            rand_means = df.groupby('model_name')['ft_test_acc'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
            test_acc = rand_means.reindex(index=natsorted(rand_means.index))
        print(test_acc - base_means.values)   # both pd.series elements
        diff = test_acc - base_means.values
        ax.plot(total, diff, label=str(label))

    plt.ylim((-20, 80))
    plt.xscale("log")
    plt.xlim((0, 100))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()

# Plot ID and SS from df for all pre-datasets
def plot_metric_all(metrics=['SS', 'ID', 'RSA']):
    pre_datasets = ['imagenet', 'places365', 'cars', 'vggface', 'segnet', 'cifar10', 'random_init']

    xticks = ['in', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc1', 'fc2', 'out']

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)
        ax.set_title(f'{metric} over VGG model layers [extract: {ft_dataset}]', weight='semibold')

        for dataset in pre_datasets:
            df_path = join(models_dir, dataset, 'df_pre_' + dataset + '+metrics')
            df = pd.read_pickle(df_path)

            # load from df
            val = df[f'{metric}_{ft_dataset}'][0]

            # since on cifar10 no extract possible on fc layers and for segnet no fc pre layers
            if dataset == 'random_init':
                val = df.groupby('model_name')[f'{metric}_{ft_dataset}'].apply(lambda g: np.mean(g.values.tolist(), axis=0))[0]
                val_std = df.groupby('model_name')[f'{metric}_{ft_dataset}'].apply(lambda g: np.std(g.values.tolist(), axis=0))[0]
                ax.fill_between(range(len(xticks)), np.array(val + 2 * val_std), np.array(val - 2 * val_std), color='pink', alpha=0.2)
            if dataset == 'segnet': val = val[:6]
            if dataset in ['segnet', 'cifar10']: x_range = range(6)
            else: x_range = range(len(xticks))

            print(dataset, x_range, val)
            ax.plot(x_range, np.array(val), '.-', label=df['model_name'][0])

        plt.ylabel(f"{metric}", weight='semibold')
        plt.xlabel("Layers", weight='semibold')
        plt.xticks(range(len(xticks)), labels=xticks)

        plt.legend(frameon=True, fancybox=True, facecolor='white', title='diff. pre datasets')  # loc="lower left",
        plt.show()


def plot_fc2_acc_id():
    pre_datasets = ['imagenet', 'places365', 'cars']  # 'vggface' only top1 # 'segnet', 'cifar10' no fc2

    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_title(f'Last hidden layer ID pred. Acc - VGG \n [extract: {ft_dataset}]', weight='semibold')

    for dataset in pre_datasets:
        df_path = join(models_dir, dataset, 'df_pre_' + dataset + '+metrics')
        df = pd.read_pickle(df_path)

        # load from df
        id = df[f'ID_{ft_dataset}'][0][7]  # for fc2
        # if dataset == 'vggface':
        #     acc = df['pre_top1'][0]
        acc = df['pre_top5'][0]
        if acc > 1: acc = acc/100

        print(dataset, id, acc)
        ax.plot((1-acc), id, 'o', label=df['model_name'][0])

    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--')

    plt.ylabel("ID of fc2", weight='semibold')
    plt.xlabel("top-1 Error", weight='semibold')

    plt.ylim((4, 10))
    plt.xlim((0, 0.2))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    plt.legend(loc="upper left", frameon=True, fancybox=True, facecolor='white', title='diff. pre datasets')
    plt.show()


if __name__ == '__main__':

    #######
    pre_dataset = 'segnet'
    ft_dataset = 'pets'
    #######

    if ft_dataset in ['custom3D', 'pets']:
        # ticks for plot - batches and epochs with bs=12 and 1200 samples
        checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '1', '3', '10', '30', '100']
        xticks = [0.0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    elif ft_dataset == 'malaria':
        # Malaria: additional _100 und _300 batches with bs=22 and 22000 samples
        checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '0_100', '0_300', '1', '3', '10', '30', '100']
        xticks = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    total = np.array(xticks)

    # plot_acc()

    # general plots over all datasets
    plot_acc_all()
    # plot_acc_all_delta()

    # plot all metrics
    # plot_metric_all(['SS', 'ID', 'RSA'])

    # plot_fc2_acc_id()
