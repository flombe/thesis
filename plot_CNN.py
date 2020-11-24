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
models_dir = join(root_dir, 'models')


# Plot ft Acc on VGG16 custom3D for different pre_datasets
def plot_acc_all():
    pre_datasets = ['mnist', 'fashionmnist', 'mnist_split1', 'mnist_split2', 'mnist_noise_struct', 'mnist_noise']

    fig1, ax1 = plt.subplots(figsize=(6, 7), dpi=200, constrained_layout=True)
    ax1.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.1)
    plt.title(f"Fine-tune Accuracies of CNN models on {ft_dataset} dataset")
    plt.xlabel("Fine-tuning Epochs (batch1 to epoch100)")
    plt.ylabel("Test Accuracy")

    colors = plt.cm.rainbow(np.linspace(0, 1, len(pre_datasets)))
    for dataset, color in zip(pre_datasets, colors):
        load_dir = join(models_dir, dataset, f'ft_{ft_dataset}')
        label = f'ft_{dataset}_{ft_dataset}'
        case = 'ft'

        # load Acc from df
        df = pd.read_pickle(join(load_dir, "df_" + label))
        # get mean and std 'ft_test_acc' over model_names for 10 seeds
        means = df.groupby('model_name')[f'{case}_test_acc'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
        stds = df.groupby('model_name')[f'{case}_test_acc'].apply(lambda g: np.std(g.values.tolist(), axis=0))
        means = means.reindex(index=natsorted(means.index))
        stds = stds.reindex(index=natsorted(stds.index))
        # print(dataset)
        # print(means, stds)
        if ft_dataset == 'fashionmnist' and dataset == 'mnist': label = label[:-5]  # naming in mnist ft is only 'fashion'

        alphas = np.linspace(0.1, 1, len(checkpts))
        for pre, alpha in zip(checkpts, alphas):
            print(pre, color, alpha)
            m = []
            s = []
            for ft in checkpts:
                m.append(means['model_' + label + '_' + pre + '_' + ft + '.pt'])
                s.append(stds['model_' + label + '_' + pre + '_' + ft + '.pt'])
            if pre == '100':
                ax1.plot(total, m, label=str(label), color=color, alpha=alpha, linewidth=0.5)  # linewidth=0.7
                ax1.fill_between(total, m + 2 * np.array(s), m - 2 * np.array(s), color=color, alpha=0.1)
            ax1.plot(total, m, color=color, alpha=alpha, linewidth=0.5)

    plt.xscale("log")
    ax1.axis([0, 100, 10, 100])
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
    ax1.xaxis.set_tick_params(which='minor', bottom=False)

    plt.legend(loc='lower right')

    # Fake a ScalarMappable so you can display a colormap, inset axes
    colors = plt.cm.binary(np.linspace(0, 1, len(checkpts)))
    cmap, norm = matplotlib.colors.from_levels_and_colors(range(len(checkpts) + 1), colors)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins1 = inset_axes(ax1, width='40%', height='4%', loc='upper left',
                        bbox_to_anchor=(0.53, 0, 1, 0.30), bbox_transform=ax1.transAxes)
    cbar = fig1.colorbar(sm, cax=axins1, orientation='horizontal', ticks=[0, 12])
    #cbar.set_label('pre-train duration', labelpad=-2)
    cbar.ax.set_title('pre-train duration', fontsize=10)
    cbar.ax.set_xticklabels(['0batch', '100ep'], fontdict={'horizontalalignment': 'center'})
    axins1.xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.show()

# Plot Acc Delta: post-ft vs. base-case  +std
def plot_acc_all_delta():
    pre_datasets = ['mnist', 'fashionmnist', 'mnist_split1', 'mnist_split2', 'mnist_noise_struct', 'mnist_noise']
    baseline = ft_dataset

    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    plt.title(f"Accuracy Delta to Baseline for CNN models on {ft_dataset} dataset")
    plt.xlabel("Fine-tuning/training Epochs (batch1 to epoch100)")
    plt.ylabel("Test Accuracy Delta")
    ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.1)

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
    print(len(base_means), len(base_stds))
    #if ft_dataset == 'mnist': total = total[1:]

    base = ax.axhline(linewidth=1.5, color='grey')
    ax.fill_between(total, 2 * np.array(base_stds), - 2 * np.array(base_stds), color=base.get_color(), alpha=0.2,
                    label='std_base')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(pre_datasets)))
    for dataset, color in zip(pre_datasets, colors):
        load_dir = join(models_dir, dataset, f'ft_{ft_dataset}')
        label = f'ft_{dataset}_{ft_dataset}'
        case = 'ft'

        # load Acc from df
        df = pd.read_pickle(join(load_dir, "df_" + label))
        # get mean and std 'ft_test_acc' over model_names for 10 seeds
        means = df.groupby('model_name')[f'{case}_test_acc'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
        stds = df.groupby('model_name')[f'{case}_test_acc'].apply(lambda g: np.std(g.values.tolist(), axis=0))
        means = means.reindex(index=natsorted(means.index))
        stds = stds.reindex(index=natsorted(stds.index))
        # print(dataset)
        # print(means, stds)
        if ft_dataset == 'fashionmnist' and dataset == 'mnist': label = label[:-5]  # naming in mnist ft is only 'fashion'

        alphas = np.linspace(0.1, 1, len(checkpts))
        for pre, alpha in zip(checkpts, alphas):
            print(pre, color, alpha)
            m = []
            s = []
            for ft in checkpts:
                m.append(means['model_' + label + '_' + pre + '_' + ft + '.pt'])
                s.append(stds['model_' + label + '_' + pre + '_' + ft + '.pt'])

            # in pre_mnist _0 model missing
            if ft_dataset == 'mnist':
                m = m[1:]
                s = s[1:]

            if pre == '100':
                ax.plot(total, m - base_means.values, label=str(label), color=color, alpha=alpha, linewidth=0.5)  # linewidth=0.7
                # ax.fill_between(total, m - base_means.values + 2 * np.array(s), m - base_means.values - 2 * np.array(s), color=color, alpha=0.1)
            ax.plot(total, m - base_means.values, color=color, alpha=alpha, linewidth=0.5)
            print(m - base_means.values)  # both pd.series elements

    # plt.ylim((-20, 80))
    plt.xscale("log")
    plt.xlim((0, 100))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot ID and SS from df for all pre-datasets
def plot_metric_all(metrics=['SS', 'ID', 'RSA']):
    pre_datasets = ['mnist', 'fashionmnist', 'mnist_split1', 'mnist_split2', 'mnist_noise_struct', 'mnist_noise']

    xticks = ['in', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc1', 'fc2', 'out']

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)
        ax.set_title(f'{metric} over VGG model layers on {ft_dataset} dataset')

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

        plt.ylabel(f"{metric}")
        plt.xlabel("Model Layers")
        plt.xticks(range(len(xticks)), labels=xticks)

        plt.legend(frameon=True, fancybox=True, facecolor='white', title='pretrained models')  # loc="lower left",
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
    pre_dataset = 'mnist'
    ft_dataset = 'fashionmnist'
    #######

    checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '0_100', '0_300', '1', '3', '10', '30', '100']
    xticks = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    total = np.array(xticks)

    ## general plots over all datasets
    # plot_acc_all()
    # plot_acc_all_delta()

    ## plot all metrics
    # plot_metric_all(['SS', 'ID', 'RSA'])

    # plot_fc2_acc_id()
