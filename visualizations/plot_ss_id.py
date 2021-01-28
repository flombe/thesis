import torch
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ss_id_plot(df, dataset='pre'):
    # load from df
    if dataset=='pre':
        pre_dataset = target_dataset = df['pre_dataset'][1]
        ID_col = 'ID_pre'
        SS_col = 'SS_pre'
    else:
        pre_dataset = df['pre_dataset'][0]
        target_dataset = df['target_dataset'][0]
        ID_col = 'ID_target'
        SS_col = 'SS_target'

    xticks = ['in', 'conv1', 'pool1', 'conv2', 'pool2', 'fc1', 'output']  # layer names
    plt.style.use('seaborn')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20c.colors)  # set color scheme

    fig, axs = plt.subplots(2, sharex=True, figsize=(7, 9), dpi=150)
    axs[0].set_title(f'ID and SumSqr over model layers \n [pretrained: {pre_dataset}, extract: {target_dataset}]',
                     weight='semibold')

    id_means = df.groupby('model_name')[ID_col].apply(lambda g: np.mean(g.values.tolist(), axis=0))
    id_stds = df.groupby('model_name')[ID_col].apply(lambda g: np.std(g.values.tolist(), axis=0))

    ss_means = df.groupby('model_name')[SS_col].apply(lambda g: np.mean(g.values.tolist(), axis=0))
    ss_stds = df.groupby('model_name')[SS_col].apply(lambda g: np.std(g.values.tolist(), axis=0))

    for i in natsorted(id_means.index):
        axs[0].plot(range(len(xticks)), id_means[i], '.-')
        axs[0].fill_between(range(len(xticks)), id_means[i] - 2 * id_stds[i], id_means[i] + 2 * id_stds[i], alpha=0.1)

        axs[1].plot(range(len(xticks)), ss_means[i], '.-', label=i)
        axs[1].fill_between(range(len(xticks)), ss_means[i] - 2 * ss_stds[i], ss_means[i] + 2 * ss_stds[i], alpha=0.1)

    axs[0].set_ylabel("Intrinsic Dimension", weight='semibold')
    plt.ylabel("SSW/TSS", weight='semibold')

    plt.xlabel("Layers", weight='semibold')
    plt.xticks(range(7), labels=xticks)

    plt.legend(loc="lower left", prop={'size': 7.5}, frameon=True, fancybox=True, facecolor='white',
               title='10 seed means')
    plt.show()


def ss_id_plot_adj(df, dataset='pre'):
    # load from df
    if dataset=='pre':
        pre_dataset = target_dataset = df['pre_dataset'][0]
        ID_col = 'ID_pre'
        SS_col = 'SS_pre'
    else:
        pre_dataset = df['pre_dataset'][0]
        target_dataset = dataset
        ID_col = f'ID_{dataset}'
        SS_col = f'SS_{dataset}'

    if df['pre_net'][0] == 'vgg16':
        xticks = ['in', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc1', 'fc2', 'out']
    else:
        xticks = ['in', 'conv1', 'pool1', 'conv2', 'pool2', 'fc1', 'output']  # layer names
    plt.style.use('seaborn')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20c.colors)  # set color scheme

    fig, axs = plt.subplots(2, sharex=True, figsize=(7, 9), dpi=150)
    axs[0].set_title(f'ID and SumSqr over model layers \n [pretrained: {pre_dataset}, extract: {target_dataset}]',
                     weight='semibold')

    id_means = df.groupby('model_name')[ID_col].apply(lambda g: np.mean(g.values.tolist(), axis=0))
    id_stds = df.groupby('model_name')[ID_col].apply(lambda g: np.std(g.values.tolist(), axis=0))

    ss_means = df.groupby('model_name')[SS_col].apply(lambda g: np.mean(g.values.tolist(), axis=0))
    ss_stds = df.groupby('model_name')[SS_col].apply(lambda g: np.std(g.values.tolist(), axis=0))

    for i in natsorted(id_means.index):
        axs[0].plot(range(len(xticks)), id_means[i], '.-')
        axs[0].fill_between(range(len(xticks)), id_means[i] - 2 * id_stds[i], id_means[i] + 2 * id_stds[i], alpha=0.1)

        axs[1].plot(range(len(xticks)), ss_means[i], '.-', label=i)
        axs[1].fill_between(range(len(xticks)), ss_means[i] - 2 * ss_stds[i], ss_means[i] + 2 * ss_stds[i], alpha=0.1)

    axs[0].set_ylabel("Intrinsic Dimension", weight='semibold')
    plt.ylabel("SSW/TSS", weight='semibold')

    plt.xlabel("Layers", weight='semibold')
    plt.xticks(range(len(xticks)), labels=xticks)

    plt.legend(loc="lower left", prop={'size': 7.5}, frameon=True, fancybox=True, facecolor='white',
               title='10 seed means')
    plt.show()



if __name__ == '__main__':
    ###
    dataset_trained = 'imagenet'
    model_folder = 'all'
    ###

    root_dir = os.getcwd()
    models_dir = join(root_dir, 'models/vgg16', dataset_trained)

    if model_folder == 'all':
        # load df
        df_path = join(models_dir, 'df_pre_' + dataset_trained + '+metrics')
        df = pd.read_pickle(df_path)

        # specify id and ss on 'pre' or 'target' data
        #ss_id_plot(df, 'pre')
        ss_id_plot_adj(df, 'custom3D')
