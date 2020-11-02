import os
from os.path import join
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from natsort import natsorted

# plot fine-tune on dataset_name on pre-trained models of 10 seeds
pre_dataset = 'mnist_split2'
ft_dataset = 'mnist'

plot_acc = False
plot_delta = True
plot_compare_switched = False
plot_compare_switched_singles = True


root_dir = os.getcwd()
models_dir = join(root_dir, 'models', pre_dataset, 'ft_' + ft_dataset)

run_name = join(f'ft_{pre_dataset}_{ft_dataset}_')  # 'ft_mnist2_mnist_'
if ft_dataset == 'fashionmnist': run_name = join(f'ft_{pre_dataset}_fashionmnist_')  # naming only fashion


checkpts = ['0_1', '0_3', '0_10', '0_30', '0_100', '0_300', '1', '3', '10', '30', '100']
xticks = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100]

x = np.array([0,1,3,10,30,100,300], dtype=int)
bs = x/937.5
ep = np.array([1,3,10,30,100], dtype=int)
total = np.append(bs, ep)

# aggregate Acc's from 10 seed runs of 12 different models in dict
mydict = dict()
for check in checkpts:
    accs = np.zeros((10, 12))
    for seed in range(1, 11):
        model_dir = join(models_dir, 'models_' + str(seed))
        train_stats = join(model_dir, run_name + check + '_train_stats.json')

        # concat batch and epoch stats for Acc
        test_acc = []
        with open(train_stats, 'r') as myfile:
            data = myfile.read()
        obj = json.loads(data)
        test_acc += obj['ft_test_acc']

        accs[seed-1] = test_acc
    mydict.update({check: accs})
#print(mydict)



### load df instead
target_model_dir = join(root_dir, 'models', ft_dataset)
if ft_dataset == 'fashionmnist':
    df = pd.read_pickle(join(target_model_dir, "df_pre_fashion"))
else:
    df = pd.read_pickle(join(target_model_dir, "df_pre_" + ft_dataset))

# get mean 'pre_test_acc' for every model_name of target task
target_means = df.groupby('model_name')['pre_test_acc'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
target_stds = df.groupby('model_name')['pre_test_acc'].apply(lambda g: np.std(g.values.tolist(), axis=0))
target_means = target_means.reindex(index=natsorted(target_means.index))
target_stds = target_stds.reindex(index=natsorted(target_stds.index))


plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Paired.colors)  # set color scheme

# Plot post-ft accuracy vs. number of training epochs
if plot_acc:
    fig1, ax1 = plt.subplots(figsize=(6, 7), dpi=150)
    plt.title(f"pre: {pre_dataset}, ft: {ft_dataset} \n Post-ft Accuracies vs. training on target (mean of 10 seeds)")
    plt.xlabel("Fine-Tuning/training Epochs (batch1 to epoch100)")
    plt.ylabel("Post-Ft/training Test Accuracy")

    for check in mydict.keys():
        accs = mydict[check]
        # for i in range(accs.shape[0]):
        #     ax1.plot(total, accs[i], 'x') #, label=(str(check) +' ft_ '+str(i)))

        mean = []
        for i in range(accs.shape[1]):
            mean.append(np.mean(accs[:,i]))
        ax1.plot(total, mean, label=(str(check)))  # +' mean'

    # plot base case: training target normaly - use mean of 10 seeds and plot std shade
    base = ax1.plot(total[1:], target_means, '-.', label='base_case', c='k')
    ax1.fill_between(total[1:], target_means - 2 * target_stds, target_means + 2 * target_stds, color=base[0].get_color(),
                     alpha=0.1, label='std_base')

    plt.ylim((0,100))
    ax1.minorticks_on()
    plt.xscale("log")
    # plt.xticks(xticks, rotation=80)
    # f = lambda x,pos: str(x).rstrip('0').rstrip('.')
    # ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(f))
    # ax1.xaxis.set_tick_params(which='minor', bottom=False)
    plt.legend(loc=4)
    plt.tight_layout()
    plt.show()


# Plot Acc Delta post-ft vs. base-case  +std
if plot_delta:
    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=150)
    plt.title(f"pre: {pre_dataset}, ft: {ft_dataset} \n Delta Post-ft Acc. - base-case Acc.")
    plt.xlabel("Fine-Tuning/training Epochs (batch1 to epoch100)")
    plt.ylabel("Post-Ft/training Test Accuracy")

    for check in mydict.keys():
        accs = mydict[check]

        mean = []
        for i in range(accs.shape[1]):
            mean.append(np.mean(accs[:, i]))
        ax2.plot(total[1:], mean[1:]-target_means, label=(str(check)))

    base = ax2.axhline(linewidth=0.5, color='k')
    # std
    ax2.fill_between(total[1:], 2 * target_stds, -2 * target_stds, color=base.get_color(),
                     alpha=0.05, label='std_base')

    plt.ylim(bottom=-5)  ##
    ax2.minorticks_on()
    plt.xscale("log")
    plt.xticks(xticks, rotation=80)
    f = lambda x,pos: str(x).rstrip('0').rstrip('.')
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(f))
    ax2.xaxis.set_tick_params(which='minor', bottom=False)
    plt.legend(loc=1)
    plt.tight_layout()
    plt.show()



# plot single plots for all pre with ft and pre-train switched
if plot_compare_switched:
    # load base case - mnist/fashionmnist with df
    base_dir = join(root_dir, 'models', ft_dataset)
    if ft_dataset == 'fashionmnist':
        df = pd.read_pickle(join(base_dir, "df_pre_fashion"))
    else:
        df = pd.read_pickle(join(base_dir, "df_pre_" + ft_dataset))

    # get mean 'pre_test_acc' for every model_name of target task
    target_means = df.groupby('model_name')['pre_test_acc'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
    target_stds = df.groupby('model_name')['pre_test_acc'].apply(lambda g: np.std(g.values.tolist(), axis=0))
    target_means = target_means.reindex(index=natsorted(target_means.index))
    target_stds = target_stds.reindex(index=natsorted(target_stds.index))

    # pre datasets
    datasets = ['mnist_noise', 'mnist_noise_struct', 'mnist_split1', 'mnist_split2', 'mnist', 'fashionmnist']

    for pre_set in datasets:
        print(pre_set)

        models_dir = join(root_dir, 'models', pre_set, 'ft_' + ft_dataset)
        run_name = join(f'ft_{pre_set}_{ft_dataset}_')  # 'ft_mnist2_mnist_'
        # if ft_dataset == 'fashionmnist': run_name = join(f'ft_{pre_dataset}_fashionmnist_')  # naming only fashion
        if ft_dataset == 'fashionmnist':
            if pre_set == 'mnist_noise_struct': run_name = join(f'ft_{pre_set}_mnist_')
            if pre_set == 'mnist': run_name = join(f'ft_{pre_set}_fashion_')

        # load the acc.values for all the presets with the json training files  (just because of legacy code)
        mydict = dict()
        # check = '1'
        for check in checkpts:
            # if check == '1':
            if pre_set == 'mnist':
                if ft_dataset == 'mnist': accs = np.zeros((10, 12))
                else: accs = np.zeros((10, 11))
            else: accs = np.zeros((10, 12))
            for seed in range(1, 11):
                model_dir = join(models_dir, 'models_' + str(seed))
                train_stats = join(model_dir, run_name + check + '_train_stats.json')

                # concat batch and epoch stats for Acc
                test_acc = []
                with open(train_stats, 'r') as myfile:
                    data = myfile.read()
                obj = json.loads(data)
                test_acc += obj['ft_test_acc']

                accs[seed - 1] = test_acc
            mydict.update({check: accs})

        lines = np.zeros((11, 11))
        std_inverted = np.zeros((11, 11))
        j = 0
        for check in mydict.keys():
            accs = mydict[check]
            mean = []
            std = []
            print(range(accs.shape[1]))
            for i in range(accs.shape[1]):  # 0,12
                print(accs[:, i])
                mean.append(np.mean(accs[:, i]))  # mean over 10 seeds
                print(mean)
                std.append(np.std(accs[:, i]))

            print(len(mean))
            if len(mean)==12:
                mean = mean[1:]
                std = std[1:]
            print(len(total), len(mean), len(target_means))
            print(len(total), mean, target_means)
            lines[j] = mean  #-target_means
            std_inverted[j] = std
            j += 1

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        plt.title(f"Pre on {pre_set} and ft on {ft_dataset}")
        plt.xlabel("Pre-training Epochs (batch1 to epoch100)")
        plt.ylabel("Accuracy")

        for i in range(11):
            print('Acc values: ', lines[:, i], ' // vs. target mean i: ', target_means[i])
            print('Diff: ', lines[:, i]-target_means[i])
            ax.plot(total[1:], lines[:, i]-target_means[i], label=str(checkpts[i]))
            ax.fill_between(total[1:], lines[:, i]-target_means[i] + 2 * np.array(std_inverted[:, i]),
                                       lines[:, i]-target_means[i] - 2 * np.array(std_inverted[:, i]), alpha=0.05)

        # ax.fill_between(total[1:], 2 * target_stds, -2 * target_stds, color='k', alpha=0.05, label='std_base')

        plt.ylim(bottom=-5)
        ax.minorticks_on()
        plt.xscale("log")
        plt.xticks(xticks, rotation=80)
        f = lambda x,pos: str(x).rstrip('0').rstrip('.')
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(f))
        ax.xaxis.set_tick_params(which='minor', bottom=False)
        plt.legend(loc=2, title='fine-tuned for')  # prop={'size': 10}
        plt.tight_layout()
        plt.show()


# plot compare switched with all pre in one plot for one ft-epoch
if plot_compare_switched_singles:
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    plt.title(f"Delta ft - base Acc. on ft: {ft_dataset}")
    plt.xlabel("Pre-training Epochs (batch1 to epoch100)")
    plt.ylabel("Accuracy")

    # load base case - mnist/fashionmnist with df
    base_dir = join(root_dir, 'models', ft_dataset)
    if ft_dataset == 'fashionmnist':
        df = pd.read_pickle(join(base_dir, "df_pre_fashion"))
    else:
        df = pd.read_pickle(join(base_dir, "df_pre_" + ft_dataset))

    # get mean 'pre_test_acc' for every model_name of target task
    target_means = df.groupby('model_name')['pre_test_acc'].apply(lambda g: np.mean(g.values.tolist(), axis=0))
    target_stds = df.groupby('model_name')['pre_test_acc'].apply(lambda g: np.std(g.values.tolist(), axis=0))
    target_means = target_means.reindex(index=natsorted(target_means.index))
    target_stds = target_stds.reindex(index=natsorted(target_stds.index))

    # pre datasets
    datasets = ['mnist_noise', 'mnist_noise_struct', 'mnist_split1', 'mnist_split2', 'mnist', 'fashionmnist']

    for pre_set in datasets:
        print(pre_set)

        models_dir = join(root_dir, 'models', pre_set, 'ft_' + ft_dataset)
        run_name = join(f'ft_{pre_set}_{ft_dataset}_')  # 'ft_mnist2_mnist_'
        # if ft_dataset == 'fashionmnist': run_name = join(f'ft_{pre_dataset}_fashionmnist_')  # naming only fashion
        if ft_dataset == 'fashionmnist':
            if pre_set == 'mnist_noise_struct': run_name = join(f'ft_{pre_set}_mnist_')
            if pre_set == 'mnist': run_name = join(f'ft_{pre_set}_fashion_')

        # load the acc.values for all the presets with the json training files  (just because of legacy code)
        mydict = dict()
        for check in checkpts:
            if pre_set == 'mnist':
                if ft_dataset == 'mnist': accs = np.zeros((10, 12))
                else: accs = np.zeros((10, 11))
            else: accs = np.zeros((10, 12))
            for seed in range(1, 11):
                model_dir = join(models_dir, 'models_' + str(seed))
                train_stats = join(model_dir, run_name + check + '_train_stats.json')

                # concat batch and epoch stats for Acc
                test_acc = []
                with open(train_stats, 'r') as myfile:
                    data = myfile.read()
                obj = json.loads(data)
                test_acc += obj['ft_test_acc']

                accs[seed - 1] = test_acc
            mydict.update({check: accs})

        lines = np.zeros((11, 11))
        std_inverted = np.zeros((11, 11))
        j = 0
        # check = '1'
        for check in mydict.keys():
        # if check == '1':
            accs = mydict[check]
            mean = []
            std = []
            for i in range(accs.shape[1]):
                mean.append(np.mean(accs[:, i]))
                std.append(np.std(accs[:, i]))

            if len(mean)==12:  # cut the 0 case, that's only available for all models
                print('len=12 cut first one for model_0')
                mean = mean[1:]
                std = std[1:]
            # print(len(total), len(mean), len(target_means))
            lines[j] = mean-target_means
            std_inverted[j] = std
            j += 1


        # pick model to plot
        model_nr = 4  ###

        ax.plot(total[1:], lines[:, model_nr], label=str(checkpts[model_nr] + ' ' + pre_set))
        ax.fill_between(total[1:], lines[:, model_nr] + 2 * np.array(std_inverted[:, model_nr]),
                                   lines[:, model_nr] - 2 * np.array(std_inverted[:, model_nr]), alpha=0.05)

    ax.fill_between(total[1:], 2 * target_stds[model_nr], -2 * target_stds[model_nr], color='k', alpha=0.08,
                    label='std_base')

    # plt.ylim(bottom=-5)
    ax.minorticks_on()
    plt.xscale("log")
    plt.xticks(xticks, rotation=80)
    f = lambda x,pos: str(x).rstrip('0').rstrip('.')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(f))
    ax.xaxis.set_tick_params(which='minor', bottom=False)
    plt.legend(loc=2, title='fine-tuned for', frameon=True, fancybox=True, facecolor='white')  # prop={'size': 10}
    plt.tight_layout()
    plt.show()
