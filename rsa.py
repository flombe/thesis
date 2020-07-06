import torchvision
import torch
import numpy as np
import scipy.stats
import sklearn.manifold
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn
import pandas as pd
import sklearn
import os
import pickle
import time
from scipy.spatial import distance
from sklearn.metrics.pairwise import manhattan_distances
import torch.nn as nn
from os.path import join
from tqdm import tqdm
from natsort import natsorted
import dill
import time
import math


def correlationd_matrix(activations):
    n = len(activations)
    correlationd = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            correlationd[i, j] = correlationd[j, i] = 1 - scipy.stats.pearsonr(activations[i],
                                                                               activations[j])[0]  #[0] for pearson coeff
    return correlationd


def calculate_activations_correlations(models, layer):

    # loaded extracted activations + labels for multiple models
    corr_distances_dict = {}
    for name, model in tqdm(natsorted(models.items())):
        print('  >> model name: ', name)
        layers = model['layers']  # input + 6 model output layers
        labels = model['labels']  # len=520, 20 activations and labels too much

        inputs = layers[layer]

        if len(labels) > 500:
            inputs = inputs[0:500, :]  # encoder last layer activations - of 500 samples
            # print(inputs.shape)  # torch.Size([500, 1600])
            labels = labels[0:500]

        corr_distances_dict[name] = correlationd_matrix(inputs)  ## corr_dist_dict = corr_matrix

    return corr_distances_dict



def dist_between_corr_matrices(dist_fun, corr_matrix_1, corr_matrix_2):
    triu_model_1 = corr_matrix_1[np.triu_indices(corr_matrix_1.shape[0], k=1)]
    triu_model_2 = corr_matrix_2[np.triu_indices(corr_matrix_2.shape[0], k=1)]

    return dist_fun(triu_model_1, triu_model_2)


def calc_rdm(dist_fun, corr_distances_dict):
    t_rdm = time.time()
    n = len(corr_distances_dict)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            rdm[i, j] = rdm[j, i] = dist_between_corr_matrices(dist_fun,
                                                               corr_distances_dict[i],
                                                               corr_distances_dict[j])
    print(f'>> Calculate {n}x{n} RDM' + '(in {:.2f} s)'.format(time.time()-t_rdm))
    return rdm


def plot_rdm(rdm):
    plt.figure(figsize=(16, 14), dpi=150)
    ax = seaborn.heatmap(rdm, cmap='rainbow')#, xticklabels=labels, yticklabels=labels)
    # n = 10  # Keeps every nth label
    # [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    # [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 0]


def multi_plot(corr_dict):
    #eval(type(corr_dict)==dict)
    n = math.sqrt(len(corr_dict))
    fig, axs = plt.subplots(math.ceil(n), math.floor(n), sharex='col', sharey='row', dpi=150)

    for ax, val in zip(axs.ravel(), corr_dict.items()):
        ax = seaborn.heatmap(val[1], cmap='rainbow') # , xticklabels=labels, yticklabels=labels)
        ax.set_title(val[0])

def multi_plottoni(corr_dict):
    fig, axs = plt.subplots(2, 6, figsize=(15, 5), edgecolor='k', sharex='col', sharey='row')
    fig.subplots_adjust(hspace=.5, wspace=.001)

    for ax, d in zip(axs.ravel(), corr_dict.items()):
        print(ax)
        ax.pcolor(d[1], cmap='rainbow')
        ax.set_title(d[0])



if __name__ == '__main__':

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Devise used = cuda on ", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Devise used = ", device)

    # load extracted activations
    root_path = os.getcwd()
    models_dir = join(root_path, 'models', 'mnist2class')
    #### TO-DO: for i in range(1, 11):  # for 10 seed folders
    load_extracted = join(root_path, 'data/mnist2class/models_1', '_extracted.pt')
    models = torch.load(load_extracted)
    print('loaded - ', load_extracted)
    print('nr of models: ', len(models))
    # print('loaded model list: ', [name for name, model in natsorted(models.items())])

    labels = list(models.values())[0]['labels'].tolist()  # same for all models - use later for plots

    path = join(models_dir, 'corr_dict_layer4.pik')
    if not os.path.exists(path):
        # on out of encoder so layer=4
        a = time.time()
        corr_dict_layer4 = calculate_activations_correlations(models, layer=4)
        print('time for corr_dict calculation: ', time.time()-a)
        # save it in models folder
        with open(str(path), 'wb') as f:
            dill.dump(corr_dict_layer4, f)
    else:
        print('already calculated, load cor_dict')
        with open(str(path), 'rb') as f:
            corr_dict_layer4 = dill.load(f)

    # # plotting
    # plt_time = time.time()
    # for model, correlation in corr_dict_layer4.items():
    #     print(model)
    #     plot_rdm(correlation)
    #     plt.title(join("1-CorrelationMatrix on 500 inputs of ", model), weight='semibold')
    #     plt.show()
    # print('time to plot all models: ', time.time()-plt_time)

    # print('alternative multi_plot_corr_dict')
    # multi_plot(corr_dict_layer4)
    # plt.title("1-CorrelationMatrix on 500 inputs of all Models", weight='semibold')
    # plt.show()

    print('multi plot example')
    multi_plottoni(corr_dict_layer4)
    plt.show()


    # corr_dict_layer4 is dict of model name and activations on layer 4
    # need only activations list
    activations_list_layer4 = list(corr_dict_layer4.values())
    print(len(activations_list_layer4))
    print(activations_list_layer4[0].shape)


    # input two corr_matrices into rdm calculation (2 corr_dist_dicts)
    print('calc and plot RDM on all models layer4 correlations')
    rdm = calc_rdm(distance.euclidean, activations_list_layer4)
    plot_rdm(rdm)
    plt.title("RMD of 1-CorrMatrices of all Models", weight='semibold')
    plt.show()




#### TO-DO

    # take extracted activations of 500 source sampels -- create RDM
    # input 500 TARGET samples (classifier maybe doesn't make sense) extract activations -- create RDM
    # -> calc correlation between these two RDMs = similarity of activations representation of different data in same NN
    # ---> therfore predicts post-ft Acc since similar representations will do similarly good?

    ## only do it on one layer?
    ## Option: compare RDMs of every layer and create score out of those (reasonable since, comparing same with same)
    ## [aggregating all layers - sinnvoll? correlation not, since layers are supposed to be different, mean also not..]


# from paper
# take from encoder output layer (so before fc layers)
# take 500 samples forwardpass activations on that layer - calculate RDM (pair-wise correlation)
# correlate these RDMs of different tasks --> in our case take the different pre-trained models?
### What is the insight of that? Low trained models corr with other low trained, and high with high?
### How does that help identify the one best in Transferability? Choose the one with highest correlation to
### a RDM of the target data? Since one wants to TL the encoder part, it's the right architecture and the
### goal is that it produces good representation on target data once TL
### -->> GOAL RSA: Do forward-pass (500 img) on the pre-trained model once with source data and once with target
### -->>> Ideally the model with highest TL shows highest correlation of RDMS
### -->>>> So, without FT every pre-trained model, we know which one will have highest post-ft Acc? (Hypothisis to check)


