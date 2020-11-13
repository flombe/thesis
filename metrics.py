import torch
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def unnesting(df, explode, axis):
    if axis == 1:
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat([pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
        df1.index = idx
        return df1.join(df.drop(explode, 1), how='left')
    else:
        df1 = pd.concat([pd.DataFrame(df[x].tolist(), index=df.index).add_prefix(x) for x in explode], axis=1)
        return df1.join(df.drop(explode, 1), how='left')


def get_values_from_df(dataset_trained, target, metr):
    metrics = pd.DataFrame()
    accs = pd.DataFrame()

    for dataset in dataset_trained:
        root_dir = os.getcwd()
        models_dir = join(root_dir, 'models', dataset)

        # load df
        df_pre = pd.read_pickle(join(models_dir, 'df_pre_' + dataset + '+metrics'))  # df 12x10 rows

        metric_mean = df_pre.groupby('model_name', sort=False)[f'{metr}_{target}'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
        # print('metric mean of 10 seeds: ', metric_mean)  # 12 x [7]

        # additional std (maybe later for adj. data)
        # metric_std = df_pre.groupby('model_name')[f'{metr}_{target}'].apply(lambda g: np.std(g.values.tolist(), axis=0))
        # print('metric std: ', metric_std.values.tolist())

        # collect all ids in one df
        metrics[dataset] = metric_mean.values  # metric_mean[:5] if leave out fc layers

        # collect all acc in one df
        df_ft = join(models_dir, f'ft_{target}', f'df_ft_{dataset}_{target}')  # 144x10
        df_ft = pd.read_pickle(df_ft)
        test_acc_mean = df_ft.groupby('model_name', sort=False)['ft_test_acc'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
        # print('ft-test Acc mean of 10 seeds: ', test_acc_mean)  # 144
        accs[dataset] = test_acc_mean.values

    print('Accuracys: ', accs)  # [144 rows x 3 columns]
    print('----- Metrics: ------ ')  # [12 rows x 3 columns] with [7] lists
    print(metrics)

    return metrics, accs
    # return metrics.transpose(), accs.transpose()

def rankcorr_and_plot(dataset_trained, target, metr):

    metrics, accs = get_values_from_df(dataset_trained, target, metr)

    # print(metrics_T)
    # print(metrics_T[0])

    # print('transponse trial')
    dff = pd.DataFrame(metrics)
    # dfff = dfff.explode('mnist_split1')
    # dfff = dfff.explode('mnist_split2')
    # dfff = dfff.explode('mnist_noise_struct')
    # print(dfff)
    # print(dfff.transpose())
    #print(dfff.iloc[0, :].apply(pd.Series.explode))
    # print(metrics.iloc[0, :])
    # print(metrics.iloc[0,:].transpose())

    checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '0_100', '0_300', '1', '3', '10', '30', '100']
    xticks = ['conv1', 'pool1', 'conv2', 'pool2', 'fc1', 'out']
    # metrics_T.rename(columns={0: f'{metr}_in', 1: f'{metr}_conv1', 2: f'{metr}_pool1', 3: f'{metr}_conv2',
    #                           4: f'{metr}_pool2', 5: f'{metr}_fc1'}, inplace=True)


    for dataset in metrics.columns:
        print(dataset)

        fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
        ax.set_title(f'Spearman rank-corr. of {metr} metric to post-ft Acc \n [extract: {target}]', weight='semibold')
        ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)

        # for all different checkpoints
        for i in range(len(checkpts)):  # 12 checkpts each with 12 values
            # append Acc's of checkpoint i
            print(dff)
            metrics_model = dff.iloc[0,:]
            print(metrics_model)
            me = pd.DataFrame(metrics_model)
            met = unnesting(me, me.columns, axis=0)
            print(met)
            accs_model = accs[dataset][i:i+len(met)]
            print(accs_model)

            df = pd.concat(met, accs_model, axis=1)
            print(df)

            ## readability
            df.rename(columns={df.columns[-1]: f'Acc{checkpts[i]}'}, inplace=True)
            df_sorted = df.sort_values(by=df.columns[-1], ascending=False)  # for rank-corr not necessary
            rank_corr = df_sorted.corr(method='spearman')  # just to see check whole corr matrix

            corr = df_sorted.corr(method="spearman").iloc[-1]
            if i < 7: a = 0.2
            else: a = 1
            ax.plot(range(len(xticks)), corr[1:], '.-', label=checkpts[i], alpha=a)  # 'in' layer not useful

        plt.ylabel("Rank-Corr Rho", weight='semibold')
        plt.xlabel("Layers", weight='semibold')
        plt.xticks(range(len(xticks)), labels=xticks)
        plt.ylim((-1, 1))

        plt.legend(frameon=True, fancybox=True, facecolor='white', title='ft duration')  # loc="lower left",
        plt.show()




def get_values_from_df_vgg(dataset_trained, target, metr):
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
            print('metric mean over seeds:', base_means[0])
            metric = base_means[0]

            # additional std (maybe later for adj. data)
            base_std = df_pre.groupby('model_name')[f'{metr}_{target}'].apply(lambda g: np.std(g.values.tolist(), axis=0))
            print('std:', base_std.values.tolist())

        # collect all ids in one df
        metrics[dataset] = abs(np.array(metric[:6]))

        # collect all acc in one df
        df_ft = join(models_dir, f'ft_{target}', f'df_ft_{dataset}_{target}')
        df_ft = pd.read_pickle(df_ft)
        accs[dataset] = df_ft['ft_test_acc']

    return metrics.transpose(), accs.transpose()


def rankcorr_and_plot_vgg(dataset_trained, target, metr):

    metrics_T, accs_T = get_values_from_df_vgg(dataset_trained, target, metr)

    checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '1', '3', '10', '30', '100']
    xticks = ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']

    # just for readability of df to check values
    metrics_T.rename(columns={0: f'{metr}_in', 1: f'{metr}_pool1', 2: f'{metr}_pool2', 3: f'{metr}_pool3',
                              4: f'{metr}_pool4', 5: f'{metr}_pool5'}, inplace=True)

    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    ax.set_title(f'Spearman rank-corr. of {metr} metric to post-ft Acc \n [extract: custom3D]', weight='semibold')
    ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)

    # for all different checkpoints
    for i in range(len(checkpts)):
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

    plt.legend(frameon=True, fancybox=True, facecolor='white', title='ft duration')  # loc="lower left",
    plt.show()





if __name__ == '__main__':

    vgg = False

    if vgg:
        dataset_trained = ['imagenet', 'places365', 'cars', 'vggface', 'segnet', 'cifar10', 'random_init']
        target = 'custom3D'
    else:
        # dataset_trained = ['mnist', 'fashionmnist', 'mnist_split1', 'mnist_split2', 'mnist_noise_struct', 'mnist_noise']  # 'cifar10'
        # target = 'mnist'
        dataset_trained = ['mnist_split1', 'mnist_split2', 'mnist_noise_struct']
        target = 'mnist'

    metrics = ['ID'] # , 'SS', 'RSA']  # set ID, SS, RSA

    for metr in metrics:
        # gets metrics and accs from dfs, calculates rank-corr to accs and plots correlation for all models
        if vgg:
            rankcorr_and_plot_vgg(dataset_trained, target, metr)
        else:
            rankcorr_and_plot(dataset_trained, target, metr)
