import torch
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
import sklearn.metrics

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
    accs = pd.DataFrame(columns=['model', 'AUC'])

    for dataset in dataset_trained:
        root_dir = os.getcwd()
        models_dir = join(root_dir, 'models', dataset)

        # load df
        df_pre = pd.read_pickle(join(models_dir, 'df_pre_' + dataset + '+metrics'))  # df 120 rows

        metric_mean = df_pre.groupby('model_name', sort=False)[f'{metr}_{target}'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
        # print('metric mean of 10 seeds: ', metric_mean)  # 12 x [7]

        # additional std (maybe later for adj. data)
        # metric_std = df_pre.groupby('model_name')[f'{metr}_{target}'].apply(lambda g: np.std(g.values.tolist(), axis=0))
        # print('metric std: ', metric_std.values.tolist())

        # collect all ids in one df
        metric_mean = pd.DataFrame(metric_mean)
        metrics = metrics.append(metric_mean)  # 72 x [7]

        # collect all Acc in one df
        df_ft = join(models_dir, f'ft_{target}', f'df_ft_{dataset}_{target}')  # 144x10
        df_ft = pd.read_pickle(df_ft)
        # df_ft = df_ft.loc[df_ft['ft_epochs'] == 100]  # only get values for 100 epoch trained (like on VGG)
        test_acc_mean = df_ft.groupby(['model_name', 'ft_epochs'], sort=False)['ft_test_acc'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
        test_acc_mean = pd.DataFrame(test_acc_mean)
        print(test_acc_mean.head())

        checkpts = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
        for i in range(len(checkpts)):
            sub_df = test_acc_mean.iloc[(i*12):((i + 1) * 12)]
            sub_name = sub_df.index[0][0]
            accs.loc[len(accs)] = [sub_name[:-5], sklearn.metrics.auc(checkpts, sub_df['ft_test_acc'])]

    print('--- Accuracys --- ', accs)  # 72 rows x 2 (model, AUC)
    print('--- Metrics: --- ', metrics)  # 72 rows, 1 column with [7] list

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
        ax.set_title(f'Spearman Corr. of {metr} metric to ft-Acc. on {target} dataset')
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

        plt.ylabel("Rank Correlation Coefficient")
        plt.xlabel("Model Layers")
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
    if target == 'malaria':
        checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '0_100', '0_300', '1', '3', '10', '30', '100']
    xticks = ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']

    # just for readability of df to check values
    metrics_T.rename(columns={0: f'{metr}_in', 1: f'{metr}_pool1', 2: f'{metr}_pool2', 3: f'{metr}_pool3',
                              4: f'{metr}_pool4', 5: f'{metr}_pool5'}, inplace=True)

    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    ax.set_title(f'Spearman Corr. of {metr} metric to ft-Acc. on {target} dataset')
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
        if i <= int(len(checkpts)/2): a = 0.2
        else: a = 1
        ax.plot(range(len(xticks)), corr[1:6], '.-', label=checkpts[i], alpha=a)  # 'in' layer not useful

    plt.ylabel("Rank Correlation Coefficient")
    plt.xlabel("Model Layers")
    plt.xticks(range(len(xticks)), labels=xticks)
    plt.ylim((-1, 1))

    plt.legend(frameon=True, fancybox=True, facecolor='white', title='ft duration')  # loc="lower left",
    plt.show()


def significance():
    # permutation
    from sympy.utilities.iterables import multiset_permutations
    import numpy as np
    from scipy import stats

    x = np.array([1, 2, 3, 4, 5, 6, 7])

    correlations = []

    for y in multiset_permutations(x):
        corr, p_value = stats.spearmanr(x, y)
        # print(corr, p_value)
        correlations.append(corr)

    print(correlations)
    print(np.percentile(correlations, 5), np.percentile(correlations, 95))
    # -0.6785714285714287, 0.6785714285714287
    print(np.percentile(correlations, 2.5), np.percentile(correlations, 97.5))
    # -0.7500000000000002, 0.7500000000000002
    print(np.percentile(correlations, 1), np.percentile(correlations, 99))
    # -0.8571428571428573, 0.8571428571428574
    print(np.percentile(correlations, 0.5), np.percentile(correlations, 99.5))
    # -0.8928571428571429, 0.8928571428571429


    # Fisher
    r = 0.9
    n = 7
    F = np.arctanh(r)
    # z-score
    z = np.sqrt((n-3)/1.06) * F
    print(z)

    # t-distribution
    t = np.sqrt((n-2)/(1-np.square(r)))
    import scipy
    print(t, scipy.stats.t(t))



if __name__ == '__main__':

    vgg = False

    if vgg:
        dataset_trained = ['imagenet', 'places365', 'cars', 'vggface', 'segnet', 'cifar10', 'random_init']
        target = 'pets'  # 'custom3D' 'malaria'
    else:
        dataset_trained = ['mnist', 'fashionmnist', 'mnist_split1', 'mnist_split2', 'mnist_noise_struct', 'mnist_noise']
        target = 'mnist'  # 'fashionmnist'

    metrics = ['ID', 'SS', 'RSA']  # , 'SS', 'RSA']  # set ID, SS, RSA

    metrics, accs = get_values_from_df(dataset_trained, target, 'ID')

    for metr in metrics:
        # gets metrics and accs from dfs, calculates rank-corr to accs and plots correlation for all models
        if vgg:
            rankcorr_and_plot_vgg(dataset_trained, target, metr)
        else:
            rankcorr_and_plot(dataset_trained, target, metr)


    # significance()

