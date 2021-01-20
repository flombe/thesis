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

def plot_pre_all():
    pre_datasets = ['mnist', 'fashionmnist', 'mnist_split1', 'mnist_split2', 'mnist_noise_struct', 'mnist_noise']

    fig1, ax1 = plt.subplots(figsize=(6, 7), dpi=200, constrained_layout=True)
    ax1.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.1)
    plt.title(f"Pre-train Accuracies of CNN models (mean of 10 seeds)")
    plt.xlabel("Pre-train Duration (batch1 to epoch100)")
    plt.ylabel("Test Accuracy")


    for dataset in pre_datasets:
        load_dir = join(models_dir, dataset)
        label = f'pre_{dataset}'
        case = 'pre'
        if dataset in ['mnist_split1', 'mnist_noise']: case = 'ft'  # wrong naming in df..

        # load Acc from df
        df = pd.read_pickle(join(load_dir, "df_" + label + '+metrics'))

        # get mean and std 'ft_test_acc' over model_names for 10 seeds
        means = df.groupby('model_name')[f'{case}_test_acc'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
        stds = df.groupby('model_name')[f'{case}_test_acc'].apply(lambda g: np.std(g.values.tolist(), axis=0))
        means = means.reindex(index=natsorted(means.index))
        stds = stds.reindex(index=natsorted(stds.index))
        # print(means)

        ax1.plot(total, means, label=str(label), linewidth=1.2)
        ax1.fill_between(total, means + 2 * np.array(stds), means - 2 * np.array(stds), alpha=0.05)

    plt.xscale("log")
    ax1.axis([0, 100, 0, 100])
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
    ax1.xaxis.set_tick_params(which='minor', bottom=False)

    plt.legend()
    plt.tight_layout()
    plt.show()


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
                ax1.plot(total, m, label=str(label), color=color, alpha=alpha, linewidth=1)  # linewidth=0.7
                ax1.fill_between(total, m + 2 * np.array(s), m - 2 * np.array(s), color=color, alpha=0.07)
            ax1.plot(total, m, color=color, alpha=alpha*0.5, linewidth=0.5)

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

def plot_acc_all_detail(checkpt='100'):
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
            if pre == checkpt:
                ax1.plot(total[5:], m[5:], label=str(label), color=color, alpha=alpha, linewidth=1)  # linewidth=0.7
                ax1.fill_between(total[5:], (m + 2 * np.array(s))[5:], (m - 2 * np.array(s))[5:], color=color, alpha=0.07)
            # ax1.plot(total, m, color=color, alpha=alpha, linewidth=0.5)

    plt.xscale("log")
    # ax1.axis([0.1, 100, 85, 100])
    plt.xlim([0.1, 100])
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
    ax1.xaxis.set_tick_params(which='minor', bottom=False)

    plt.legend() # loc='lower right')
    plt.tight_layout()
    plt.show()


# Plot Acc Delta: post-ft vs. base-case  +std
def plot_acc_all_delta():
    pre_datasets = ['mnist', 'fashionmnist', 'mnist_split1', 'mnist_split2', 'mnist_noise_struct', 'mnist_noise']
    baseline = ft_dataset

    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    plt.title(f"Accuracy Delta to Baseline for CNN models on {ft_dataset} dataset")
    plt.xlabel("Fine-tuning Epochs (batch1 to epoch100)")
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
    total_plot = total
    if ft_dataset == 'mnist': total_plot = total[1:]

    base = ax.axhline(linewidth=1.5, color='grey')
    ax.fill_between(total_plot, 2 * np.array(base_stds), - 2 * np.array(base_stds), color=base.get_color(), alpha=0.1,
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
                ax.plot(total_plot, m - base_means.values, label=str(label), color=color, alpha=alpha, linewidth=0.7)  # linewidth=0.7
                # ax.fill_between(total, m - base_means.values + 2 * np.array(s), m - base_means.values - 2 * np.array(s), color=color, alpha=0.1)
            ax.plot(total_plot, m - base_means.values, color=color, alpha=alpha*0.5, linewidth=0.5)
            print(m - base_means.values)  # both pd.series elements

    # plt.ylim((-20, 80))
    plt.xscale("log")
    plt.xlim((0, 100))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
    ax.xaxis.set_tick_params(which='minor', bottom=False)

    plt.legend()
    if ft_dataset == 'fashionmnist': plt.legend(loc='upper right', prop={'size': 9})

    # Fake a ScalarMappable so you can display a colormap, inset axes
    colors = plt.cm.binary(np.linspace(0, 1, len(checkpts)))
    cmap, norm = matplotlib.colors.from_levels_and_colors(range(len(checkpts) + 1), colors)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins1 = inset_axes(ax, width='40%', height='2%', loc='upper left',
                        bbox_to_anchor=(0.53, 0, 1, 0.70), bbox_transform=ax.transAxes)
    cbar = fig.colorbar(sm, cax=axins1, orientation='horizontal', ticks=[0, 12])
    cbar.ax.set_title('pre-train duration', fontsize=10)
    cbar.ax.set_xticklabels(['0batch', '100ep'], fontdict={'horizontalalignment': 'center'}, fontsize=10)
    axins1.xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.show()

# Plot ID and SS from df for all pre-datasets
def plot_metric_all(metrics=['SS', 'ID', 'RSA']):
    pre_datasets = ['mnist', 'fashionmnist', 'mnist_split1', 'mnist_split2', 'mnist_noise_struct', 'mnist_noise']

    xticks = ['in', 'conv1', 'pool1', 'conv2', 'pool2', 'fc1', 'out']

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)
        ax.set_title(f'{metric} over CNN model layers on {ft_dataset} dataset')

        for dataset in pre_datasets:
            load_dir = join(models_dir, dataset)
            label = f'pre_{dataset}'

            # load Acc from df
            df = pd.read_pickle(join(load_dir, "df_" + label + '+metrics'))

            # get mean and std 'ft_test_acc' over model_names for 10 seeds
            means = df.groupby('model_name')[f'{metric}_{ft_dataset}'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
            stds = df.groupby('model_name')[f'{metric}_{ft_dataset}'].apply(lambda g: np.std(g.values.tolist(), axis=0))
            means = means.reindex(index=natsorted(means.index))
            stds = stds.reindex(index=natsorted(stds.index))

            if dataset == 'fashionmnist': label = label[:-5]  # naming in mnist ft is only 'fashion'

            for pre in checkpts:
                m = means['model_' + label + '_' + pre + '.pt']
                s = stds['model_' + label + '_' + pre + '.pt']
                if pre == '100':
                    ax.plot(xticks, m, '.-', label=str(f'model_{label}_{pre}'), linewidth=1)  # linewidth=0.7
                    ax.fill_between(xticks, m + 2 * np.array(s), m - 2 * np.array(s), alpha=0.07)
                # ax.plot(total, m, color=color, linewidth=0.5)

        plt.ylabel(f"{metric}")
        plt.xlabel("Model Layers")
        plt.xticks(range(len(xticks)), labels=xticks)

        plt.legend(frameon=True, fancybox=True, facecolor='white', title='pretrained models')  # loc="lower left",
        plt.show()


if __name__ == '__main__':

    #######
    pre_dataset = 'mnist'
    ft_dataset = 'fashionmnist'  #'fashionmnist'
    #######

    checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '0_100', '0_300', '1', '3', '10', '30', '100']
    xticks = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    total = np.array(xticks)

    plot_pre_all()

    ## general plots over all datasets
    plot_acc_all()
    plot_acc_all_detail('100')
    plot_acc_all_delta()

    # plot all metrics
    #plot_metric_all(['SS', 'ID', 'RSA'])
