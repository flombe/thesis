import torch
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # on simple CNN
    # dataset_trained = ['mnist', splits etc. , 'imagenet', 'cifar10']

    # vgg16
    dataset_trained = ['imagenet', 'places365', 'cars', 'vggface', 'segnet', 'cifar10', 'random_init']
    target = 'custom3D'

    metr = 'ID'  ### set ID, SS, RSA

    #######

    metrics = pd.DataFrame()
    accs = pd.DataFrame()

    for dataset in dataset_trained:
        root_dir = os.getcwd()
        models_dir = join(root_dir, 'models', 'vgg16', dataset)

        # load df
        df_pre = join(models_dir, 'df_pre_' + dataset + '+metrics')
        df_pre = pd.read_pickle(df_pre)

        metric = df_pre[f'{metr}_{target}'][0]
        if len(df_pre) > 1:  # for random_init multiple seeds
            base_means = df_pre.groupby('model_name')[f'{metr}_{target}'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
            print(base_means)
            # additional std (maybe later for adj. data)
            base_std = df_pre.groupby('model_name')[f'{metr}_{target}'].apply(lambda g: np.std(g.values.tolist(), axis=0))
            print(base_std.values.tolist())
        # collect all ids in one df
        metrics[dataset] = abs(np.array(metric[:6]))

        # collect all acc in one df
        df_ft = join(models_dir, f'ft_{target}', f'df_ft_{dataset}_{target}')
        df_ft = pd.read_pickle(df_ft)
        accs[dataset] = df_ft['ft_test_acc']

    metrics_T = metrics.transpose()
    accs_T = accs.transpose()

    checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '1', '3', '10', '30', '100']
    xticks = ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']
    # plt.style.use('seaborn')
    # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20c.colors)  # set color scheme
    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    ax.set_title(f'Spearman rank-corr. of {metr} metric to post-ft Acc \n [extract: custom3D]', weight='semibold')
    ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)

    # for all different checkpoints
    for i in range(len(checkpts)):
        ## just for readability of df to check values
        metrics_T.rename(columns={0: f'{metr}_in', 1: f'{metr}_pool1', 2: f'{metr}_pool2', 3: f'{metr}_pool3',
                                  4: f'{metr}_pool4', 5: f'{metr}_pool5'}, inplace=True)

        # append Acc's of checkpoint i
        df = pd.concat([metrics_T, accs_T.iloc[:, i]], axis=1)

        ## readability
        df.rename(columns={df.columns[-1]: f'Acc{checkpts[i]}'}, inplace=True)
        df_sorted = df.sort_values(by=df.columns[-1], ascending=False)  # for rank-corr not necessary
        rank_corr = df_sorted.corr(method='spearman')  # just to see check whole corr matrix

        corr = df_sorted.corr(method="spearman").iloc[-1]
        if i < 5: a = 0.2
        else: a=1
        ax.plot(range(len(xticks)), corr[1:6], '.-', label=checkpts[i], alpha=a)  # 'in' layer not useful

    plt.ylabel("Rank-Corr Rho", weight='semibold')
    plt.xlabel("Layers", weight='semibold')
    plt.xticks(range(len(xticks)), labels=xticks)
    plt.ylim((-1, 1))

    plt.legend(loc="lower left", frameon=True, fancybox=True, facecolor='white', title='ft duration')
    plt.show()
