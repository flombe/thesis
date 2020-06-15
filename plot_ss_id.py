import torch
import datasets
import os
from os.path import join
import numpy as np
import json
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cp=['0batch1','0batch3','0batch10','0batch30','0batch100','0batch300','1','3','10','30','100']

def json_load(path):
    with open(path, 'r') as myfile:
        data = myfile.read()
    #print(list(json.loads(data).keys()))
    return json.loads(data)  #dict of models and value = ('model checkpoint', [ss for 7 layers], [id for 7 layers])


def model_select(data, model_name, checkpts=cp):
    if checkpts!=cp:
        plot_list = []
        for name in model_name:
            model = join('model_ft_mnist2_mnist_' + name)
            for i in range(11):
                if str(data[model][i][0]) in [join(model+'_'+check+'.pt') for check in checkpts]:
                    plot_list.append(data[model][i])
    else:
        plot_list = []
        print(data)
        for name in model_name:
            plot_list += data[join('model' + name)]
            #plot_list += data[join('model_ft_mnist2_mnist_' + name)]
    return plot_list



def ss_id_single_plot(path, model_name):
    print('---- Single Plot ----')
    if model_folder == 'all':
        data_all = []
        for path in paths:
            data_all.append(json_load(path))
        plot_list = []
        for data in data_all:
            plot_list += model_select(data, model_name)

        new = np.array(plot_list)
        plot_list=[]
        for i in range(11):
            plot_list.append([new[i,0],
                              [np.mean([*([np.asarray(l) for l in new[i:110:11,1]][m][k] for m in range(10))]) for k in range(7)],
                              [np.mean([*([np.asarray(l) for l in new[i:110:11,2]][m][k] for m in range(10))]) for k in range(7)]
                              ])
        print('- means of 10 seeds')
    else:
        data = json_load(path)
        print(data)
        plot_list = model_select(data, model_name)


    for line in list(plot_list):
        print(line[0])

    xticks = ['in', 'conv1', 'pool1', 'conv2','pool2', 'fc1', 'output']
    plt.style.use('seaborn')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20c.colors) #set color scheme

    fig, axs = plt.subplots(2, sharex=True, figsize=(7, 9), dpi=150)
    axs[0].set_title(join('ID and SumSqr over ft model layers [pretrained_' + str(model_name) +']'), weight='semibold')
    #axs[0].set_ylim((4, 24))

    for name, ss, ids in plot_list:
        axs[0].plot(range(len(ids)), ids, '.-')
    axs[0].set_ylabel("Intrinsic Dimension", weight='semibold')

    for name, ss, ids in plot_list:
        axs[1].plot(range(len(ss)), ss, '.-', label=name)

    plt.xlabel("Layers", weight='semibold')
    plt.xticks(range(7), labels=xticks)
    #plt.ylim((0.045, 0.1))
    plt.ylabel("SSW/TSS", weight='semibold')
    if model_folder == 'all': plt.legend(loc="lower left", prop={'size': 7.5}, frameon=True, fancybox=True, facecolor='white', title='10 seed means')
    else: plt.legend(loc="lower left", prop={'size': 7.5}, frameon=True, fancybox=True, facecolor='white')
    plt.show()




def ss_id_multi_plot(path, model_name, checkpts=cp):
    print('---- Multi Plot ----')
    if model_folder == 'all':
        data_all = []
        for path in paths:
            data_all.append(json_load(path))
        plot_list = []
        for data in data_all:
            plot_list += model_select(data, model_name, checkpts)

        new = np.array(plot_list)
        #print(new.shape)
        plot_list = []
        stats = []
        n = len(checkpts)*len(model_name)
        for i in range(n):
            plot_list.append([new[i, 0],
                              [np.mean([*([np.asarray(l) for l in new[i:(n*10):n, 1]][m][k] for m in range(10))]) for k in range(7)],
                              [np.mean([*([np.asarray(l) for l in new[i:(n*10):n, 2]][m][k] for m in range(10))]) for k in range(7)]
                              ])

            # additional statistical stats
            stats.append([new[i, 0],
                              [np.std([*([np.asarray(l) for l in new[i:(n * 10):n, 1]][m][k] for m in range(10))]) for
                               k in range(7)],
                              [np.std([*([np.asarray(l) for l in new[i:(n * 10):n, 2]][m][k] for m in range(10))]) for
                               k in range(7)]
                              ])

        print('- means of 10 seeds')
    else:
        data = json_load(path)
        print(data)
        plot_list = model_select(data, model_name)


    for line in list(plot_list):
        print(line[0])

    xticks = ['in', 'conv1', 'pool1', 'conv2','pool2', 'fc1', 'output']
    plt.style.use('seaborn')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20c.colors) #set color scheme

    fig, axs = plt.subplots(2, sharex=True, figsize=(7, 9), dpi=150)
    #axs[0].set_title(join('ID & SumSqr over ft model layers  [pretrained_' + str(model_name)+']'), weight='semibold')
    #axs[0].set_ylim((4, 24))

    i = 0
    for name, ss, ids in plot_list:
        if i%2: axs[0].plot(range(len(ids)), ids, '.--')
        else: axs[0].plot(range(len(ids)), ids, '.-')
        axs[0].fill_between(range(len(ids)), np.asarray(ids) - 2*np.asarray(stats[i][2]),
                            np.asarray(ids) + 2*np.asarray(stats[i][2]), alpha=0.1)
        i += 1
    axs[0].set_ylabel("Intrinsic Dimension", weight='semibold')

    i = 0
    for name, ss, ids in plot_list:
        axs[1].plot(range(len(ss)), ss, '.-', label=name)
        axs[1].fill_between(range(len(ss)), np.asarray(ss) - 2*np.asarray(stats[i][1]),
                            np.asarray(ss) + 2*np.asarray(stats[i][1]), alpha=0.1)
        i += 1
    plt.ylabel("SSW/TSS", weight='semibold')
    plt.ylim((0.045, 0.1))

    plt.xlabel("Layers", weight='semibold')
    plt.xticks(range(7), labels=xticks)
    if model_folder == 'all': plt.legend(loc="lower left", prop={'size': 7.5}, frameon=True, fancybox=True, facecolor='white', title='10 seed means')
    else: plt.legend(loc="lower left", prop={'size': 7.5}, frameon=True, fancybox=True, facecolor='white')
    plt.show()




if __name__ == '__main__':

    ###
    dataset_name = 'mnist'
    model_folder = 'all'  # 'all' or 'nr'
    model_name = ['100']
    ###


    root_dir = os.getcwd()
    dataset_dir = join(root_dir, 'data', dataset_name)

    if model_folder == 'all':
        paths = []
        for seed in range(1,11):
            paths.append(join(dataset_dir, 'models_' + str(seed), 'ss_id.json')) #list of paths
        ss_id_single_plot(paths, model_name=['_pre_mnist2'])
        #ss_id_multi_plot(paths, model_name)
        #ss_id_multi_plot(paths, model_name=['0batch1', '0batch10', '0batch100', '0batch300', '1', '3', '10', '100'], checkpts=['0batch300', '100'])
    else:
        path = join(dataset_dir, 'models_' + model_folder, 'ss_id.json')
        ss_id_single_plot(path, model_name=model_name)
