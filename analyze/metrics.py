import torch
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
import sklearn.metrics
import scipy.stats

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

def spearmanr_pval(x, y):
    return scipy.stats.spearmanr(x, y)[1]

def get_values_from_df(dataset_trained, target, metr):
    metrics = pd.DataFrame()
    accs = pd.DataFrame(columns=['model', 'AUC'])

    for dataset in dataset_trained:
        root_dir = os.getcwd()
        models_dir = join(root_dir, '../models', dataset)

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

        # get log-AreaUnderCurve for fine-tune Accuracy
        checkpts = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
        log_checks = range(len(checkpts))
        for i in range(len(checkpts)):
            sub_df = test_acc_mean.iloc[(i*12):((i + 1) * 12)]
            sub_name = sub_df.index[0][0]
            accs.loc[len(accs)] = [sub_name[:-5], sklearn.metrics.auc(log_checks, sub_df['ft_test_acc'])]

    print('--- Accuracys --- ', accs)  # 72 rows x 2 (model, AUC)
    print('--- Metrics: --- ', metrics)  # 72 rows, 1 column with [7] list

    return metrics, accs
    # return metrics.transpose(), accs.transpose()

def rankcorr_and_plot(dataset_trained, target, metr):
    # load metrics and accs from dfs
    metrics, accs = get_values_from_df(dataset_trained, target, metr)

    dff = unnesting(metrics, metrics.columns, axis=0)  # explode list in column into multiple columns
    dff = dff.drop(f'{metr}_{target}0', axis=1)
    dff['AUC'] = accs['AUC'].values  # append AUC ft-Acc values on df (same sorting for both)

    checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '0_100', '0_300', '1', '3', '10', '30', '100']
    xticks = ['conv1', 'pool1', 'conv2', 'pool2', 'fc1', 'out']

    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    ax.set_title(f'Spearman Corr. of {metr} metric to ft-Acc. on {target} dataset')
    ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)

    rank_corr = dff.corr(method="spearman")
    corr = dff.corr(method="spearman").iloc[-1]
    ax.plot(range(len(xticks)), corr[:-1], '.-', label='all models')  # last value is corr with itself

    # print(fisher_95int(corr[:-1].values, num=72))  # check 95 significance interval
    low, up = fisher_95int(corr[:-1].values, num=72)
    ax.fill_between(range(len(xticks)), low, up, alpha=0.9, color='whitesmoke')
    plt.hlines(low, 0, 5, color='lightgrey', alpha=0.3, linestyle=(0, (5, 10)), linewidth=1)
    plt.hlines(up, 0, 5, color='lightgrey', alpha=0.3, linestyle=(0, (5, 10)), linewidth=1)

    p_vals = dff.corr(method=spearmanr_pval).iloc[-1]
    print(f'-- {metr} on all {target} P-values --\n', p_vals[:-1])

    plt.ylabel("Rank Correlation Coefficient")
    plt.xlabel("Model Layers")
    plt.xticks(range(len(xticks)), labels=xticks)
    plt.ylim((-1, 1))
    plt.legend(frameon=True, fancybox=True, facecolor='white')  # loc="lower left",
    plt.show()




    # split data and plot for every pre-trained dataset the rank-corr on only 12 models
    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    ax.set_title(f'Spearman Corr. of {metr} metric to ft-Acc. on {target} dataset')
    ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)

    # for all different checkpoints
    for i in range(len(dataset_trained)):
        sub_df = dff.iloc[i*12:(i+1)*12, :]  # ever dataset with 12 rows
        rank_corr = sub_df.corr(method='spearman')  # just to see check whole corr matrix

        p_vals = sub_df.corr(method=spearmanr_pval).iloc[-1]
        print(f'-- {metr} on {target} of {dataset_trained[i]} P-values --\n', p_vals[:-1])

        corr = sub_df.corr(method="spearman").iloc[-1]
        ax.plot(range(len(xticks)), corr[:-1], '.-', label=dataset_trained[i])  # 'in' layer not useful
        low, up = fisher_95int(corr[:-1].values, num=12)
        ax.fill_between(range(len(xticks)), low, up, alpha=0.5, color='whitesmoke')  # print 95 conf_interval
        plt.hlines(low, 0, 5, color='lightgrey', alpha=0.3, linestyle=(0, (5, 10)), linewidth=1)
        plt.hlines(up, 0, 5, color='lightgrey', alpha=0.3, linestyle=(0, (5, 10)), linewidth=1)

    plt.ylabel("Rank Correlation Coefficient")
    plt.xlabel("Model Layers")
    plt.xticks(range(len(xticks)), labels=xticks)
    plt.ylim((-1, 1))
    plt.legend(frameon=True, fancybox=True, facecolor='white')  # loc="lower left",
    plt.show()



    # split data and plot for every pre-trained checkpoint
    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    ax.set_title(f'Spearman Corr. of {metr} metric to ft-Acc. on {target} dataset')
    ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)

    # for all different checkpoints
    for i in range(len(checkpts)):
        sub_df = dff.iloc[i::12, :]  # ever 12th row
        # print(sub_df)
        rank_corr = sub_df.corr(method='spearman')  # just to see check whole corr matrix

        p_vals = sub_df.corr(method=spearmanr_pval).iloc[-1]
        print(f'-- {metr} on {target} of {checkpts[i]} P-values --\n', p_vals[:-1])

        corr = sub_df.corr(method="spearman").iloc[-1]
        # print(fisher_95int(corr[:-1].values, num=12))  # check 95 significance interval
        if i < 7: a = 0.2
        else: a = 1
        ax.plot(range(len(xticks)), corr[:-1], '.-', label=checkpts[i], alpha=a)

        conv_intervals = permutation_significance(len(dataset_trained))
        # for lower, upper in conv_intervals:
        #     ax.fill_between(range(len(xticks)), -1, lower, color='whitesmoke', alpha=0.005)
        #     ax.fill_between(range(len(xticks)), upper, 1, color='whitesmoke', alpha=0.005)
        lower, upper = conv_intervals[2]
        ax.fill_between(range(len(xticks)), lower, upper, color='whitesmoke', alpha=0.5, linestyle='--')
        plt.hlines(lower, 0, 5, color='lightgrey', alpha=0.3, linestyle=(0, (5, 10)), linewidth=1)
        plt.hlines(upper, 0, 5, color='lightgrey', alpha=0.3, linestyle=(0, (5, 10)), linewidth=1)

        ''' significance
        90% CI: -0.7714285714285715 0.7714285714285715
        95% CI: -0.8285714285714287 0.8285714285714287
        98% CI: -0.8857142857142859 0.8857142857142858
        99% CI: -0.9428571428571431 0.9428571428571431
        '''

    plt.ylabel("Rank Correlation Coefficient")
    plt.xlabel("Model Layers")
    plt.xticks(range(len(xticks)), labels=xticks)
    plt.ylim((-1, 1))
    if metr=='RSA': plt.legend(frameon=True, fancybox=True, facecolor='white', title='pre-training', loc="lower right")
    else: plt.legend(frameon=True, fancybox=True, facecolor='white', title='pre-training', loc="upper left")
    plt.show()





    # # split data and plot for every pre-trained dataset the rank-corr on only 12 models
    # for dataset in dataset_trained:
    #     print(dataset)
    #
    #     fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    #     ax.set_title(f'Spearman Corr. of {metr} metric to ft-Acc. on {target} dataset')
    #     ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)
    #
    #     # for all different checkpoints
    #     for i in range(len(dataset_trained)):
    #         sub_df = dff.iloc[i*12:(i+1)*12, :]  # ever dataset with 12 rows
    #         print(sub_df)
    #         rank_corr = sub_df.corr(method='spearman')  # just to see check whole corr matrix
    #
    #         corr = sub_df.corr(method="spearman").iloc[-1]
    #         if i < 7: a = 0.2
    #         else: a = 1
    #         ax.plot(range(len(xticks)), corr[:-1], '.-', label=checkpts[i], alpha=a)  # 'in' layer not useful
    #
    #     plt.ylabel("Rank Correlation Coefficient")
    #     plt.xlabel("Model Layers")
    #     plt.xticks(range(len(xticks)), labels=xticks)
    #     plt.ylim((-1, 1))
    #     plt.legend(frameon=True, fancybox=True, facecolor='white', title='ft duration')  # loc="lower left",
    #     plt.show()


def rankcorr_and_plot_pool2(dataset_trained, target, metrics):
    checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '0_100', '0_300', '1', '3', '10', '30', '100']

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.set_title(f'Spearman Corr. of metrics on pool2 layer to ft-Acc. on {target} dataset')
    ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)

    for metr in metrics:
        # load metrics and accs from dfs
        metrics, accs = get_values_from_df(dataset_trained, target, metr)

        dff = unnesting(metrics, metrics.columns, axis=0)  # explode list in column into multiple columns
        dff = dff.drop([f'{metr}_{target}0', f'{metr}_{target}1', f'{metr}_{target}2', f'{metr}_{target}3',
                        f'{metr}_{target}5', f'{metr}_{target}6'], axis=1)
        dff['AUC'] = accs['AUC'].values  # append AUC ft-Acc values on df (same sorting for both)

        corr_pool2 = []
        for i in range(len(checkpts)):
            sub_df = dff.iloc[i::12, :]  # ever 12th row

            p_vals = sub_df.corr(method=spearmanr_pval).iloc[-1]
            print(f'-- {metr} on {target} of {checkpts[i]} P-values --\n', p_vals[:-1])

            corr = sub_df.corr(method="spearman").iloc[-1]
            corr_pool2.append(corr[:-1])

        ax.plot(range(len(checkpts)), corr_pool2, '.-', label=str(metr))

    # # fisher intervals
    # lower, upper = fisher_95int(corr_pool2, 6)

    conv_intervals = permutation_significance(len(dataset_trained))
    lower, upper = conv_intervals[2]
    ax.fill_between(range(len(checkpts)), lower, upper, color='whitesmoke', alpha=0.5, linestyle='--')
    plt.hlines(lower, 0, 11, color='lightgrey', alpha=0.8, linestyle=(0, (5, 10)), linewidth=1)
    plt.hlines(upper, 0, 11, color='lightgrey', alpha=0.8, linestyle=(0, (5, 10)), linewidth=1)

    ''' significance
    90% CI: -0.7714285714285715 0.7714285714285715
    95% CI: -0.8285714285714287 0.8285714285714287  *
    98% CI: -0.8857142857142859 0.8857142857142858
    99% CI: -0.9428571428571431 0.9428571428571431
    '''

    plt.ylabel("Rank Correlation Coefficient")
    plt.xlabel("pre-train checkpoints")
    plt.xticks(range(len(checkpts)), labels=checkpts)
    plt.ylim((-1, 1))
    plt.legend(frameon=True, fancybox=True, facecolor='white', title='metric')  # loc="lower left",
    plt.show()


def get_values_from_df_vgg(dataset_trained, target, metr):
    metrics = pd.DataFrame()
    accs = pd.DataFrame()

    for dataset in dataset_trained:
        root_dir = os.getcwd()
        models_dir = join(root_dir, '../models', 'vgg16', dataset)

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

        low, up = fisher_95int(corr[1:6].values, num=12)
        ax.fill_between(range(len(xticks)), low, up, alpha=0.5, color='whitesmoke')  # print 95 conf_interval
        plt.hlines(low, 0, 4, color='lightgrey', alpha=0.3, linestyle=(0, (5, 10)), linewidth=1)
        plt.hlines(up, 0, 4, color='lightgrey', alpha=0.3, linestyle=(0, (5, 10)), linewidth=1)

    plt.ylabel("Rank Correlation Coefficient")
    plt.xlabel("Model Layers")
    plt.xticks(range(len(xticks)), labels=xticks)
    plt.ylim((-1, 1))

    plt.legend(frameon=True, fancybox=True, facecolor='white', title='ft duration')  # loc="lower left",
    plt.show()




    # bundled rank corr plot

    accs = pd.DataFrame(columns=['model', 'AUC'])
    # get log-AreaUnderCurve for fine-tune Accuracy
    if target == "malaria": checkpts = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    else: checkpts = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    log_checks = range(len(checkpts))
    for i in range(7):
        sub_df = accs_T.iloc[i]
        sub_name = accs_T.index[i]
        accs.loc[len(accs)] = [sub_name, sklearn.metrics.auc(log_checks, sub_df)]

    dff = metrics_T
    dff['AUC'] = accs['AUC'].values  # append AUC ft-Acc values on df (same sorting for both)

    xticks = ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']

    fig, ax = plt.subplots(figsize=(6, 7), dpi=150)
    ax.set_title(f'Spearman Corr. of {metr} metric to ft-Acc. on {target} dataset')
    ax.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.08)

    corr = dff.corr(method="spearman").iloc[-1]
    print(corr[1:6], range(len(xticks)))
    ax.plot(range(len(xticks)), corr[1:6], '.-', label='all models')  # last value is corr with itself

    # print(fisher_95int(corr[:-1].values, num=7))  # check 95 significance interval
    low, up = fisher_95int(corr[1:6].values, num=7)
    ax.fill_between(range(len(xticks)), low, up, alpha=0.9, color='whitesmoke')
    plt.hlines(low, 0, 4, color='lightgrey', alpha=0.3, linestyle=(0, (5, 10)), linewidth=1)
    plt.hlines(up, 0, 4, color='lightgrey', alpha=0.3, linestyle=(0, (5, 10)), linewidth=1)

    p_vals = dff.corr(method=spearmanr_pval).iloc[-1]
    print(f'-- {metr} on all {target} P-values --\n', p_vals[:-1])

    plt.ylabel("Rank Correlation Coefficient")
    plt.xlabel("Model Layers")
    plt.xticks(range(len(xticks)), labels=xticks)
    plt.ylim((-1, 1))
    plt.legend(frameon=True, fancybox=True, facecolor='white')  # loc="lower left",
    plt.show()


def permutation_significance(n):
    # permutation
    from sympy.utilities.iterables import multiset_permutations
    from scipy import stats

    # x = np.array([1, 2, 3, 4, 5, 6, 7])
    x = np.array(range(0, n))

    correlations = []
    conv_intervals = []

    for y in multiset_permutations(x):
        corr, p_value = stats.spearmanr(x, y)
        # print(corr, p_value)
        correlations.append(corr)

    conv_intervals.append((np.percentile(correlations, 5), np.percentile(correlations, 95)))
    # -0.6785714285714287, 0.6785714285714287
    conv_intervals.append((np.percentile(correlations, 2.5), np.percentile(correlations, 97.5)))
    # -0.7500000000000002, 0.7500000000000002
    conv_intervals.append((np.percentile(correlations, 1), np.percentile(correlations, 99)))
    # -0.8571428571428573, 0.8571428571428574
    conv_intervals.append((np.percentile(correlations, 0.5), np.percentile(correlations, 99.5)))
    # -0.8928571428571429, 0.8928571428571429

    return conv_intervals

    # # Fisher Wikipedia
    # r = 0.9
    # n = 7
    # F = np.arctanh(r)
    # # z-score
    # z = np.sqrt((n-3)/1.06) * F
    # print(z)
    #
    # # t-distribution
    # t = np.sqrt((n-2)/(1-np.square(r)))
    # import scipy
    # print(t, scipy.stats.t(t))

# Fisher intervall
def fisher_95int(corr, num):
    # https://stats.stackexchange.com/questions/18887/how-to-calculate-a-confidence-interval-for-spearmans-rank-correlation
    import math
    low = []
    up = []
    for i in range(len(corr)):
        r = corr[i]
        r = 0  # change calc. -- since null hypothesis is r = 0
        stderr = 1.0 / math.sqrt(num - 3)
        delta = 1.96 * stderr
        lower = math.tanh(math.atanh(r) - delta)
        upper = math.tanh(math.atanh(r) + delta)
        print("lower %.6f upper %.6f" % (lower, upper))
        low.append(lower)
        up.append(upper)
    return low, up


if __name__ == '__main__':

    vgg = True

    if vgg:
        dataset_trained = ['imagenet', 'places365', 'cars', 'vggface', 'segnet', 'cifar10', 'random_init']
        target = 'pets'  # 'custom3D' 'malaria' 'pets'
    else:
        dataset_trained = ['mnist', 'fashionmnist', 'mnist_split1', 'mnist_split2', 'mnist_noise_struct', 'mnist_noise']
        target = 'mnist'  # 'fashionmnist'

    metrics = ['ID', 'SS', 'RSA']  # , 'SS', 'RSA']  # set ID, SS, RSA

    for metr in metrics:
        # gets metrics and accs from dfs, calculates rank-corr to accs and plots correlation for all models
        if vgg:
            rankcorr_and_plot_vgg(dataset_trained, target, metr)
        else:
            rankcorr_and_plot(dataset_trained, target, metr)

    # rankcorr_and_plot_pool2(dataset_trained, target, metrics)