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


def calculate_activations_correlations(models, layer, sorted=False):
    # loaded extracted activations + labels for multiple models
    corr_distances_dict = {}
    for name, model in tqdm(natsorted(models.items())):
        print('  >> model name: ', name)
        labels = model['labels']
        layers = model['layers']  # input + 6 model output layers
        inputs = layers[layer]

        if sorted==True:
            inputs = inputs[np.array(labels).argsort(),:]

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


def plot_rdm(rdm, model_names):
    fig = plt.figure(figsize=(16, 14), dpi=150)
    ax = seaborn.heatmap(rdm, cmap='rainbow', annot=True, fmt='6.3g', xticklabels=model_names, yticklabels=model_names)
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    # n = 10  # Keeps every nth label
    # [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    # [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 0]


def multi_plot_rdm(corr_dict, labels):
    n = math.sqrt(len(corr_dict))
    fig, axs = plt.subplots(math.ceil(n), math.floor(n), figsize=(16, 21), sharex='col', sharey='row', dpi=250)
    fig.subplots_adjust(hspace=.5, wspace=.001)

    for ax, val in tqdm(zip(axs.ravel(), corr_dict.items())):
        seaborn.set(font_scale=0.8)
        seaborn.heatmap(val[1], cmap='rainbow', ax=ax, cbar=True, xticklabels=labels, yticklabels=labels)
        ax.set_title(val[0], weight='semibold')

        ax.tick_params(axis='both', labelsize=3, width=0.3, length=1)
        n = 10  # Keeps every nth label
        [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 0]
    fig.delaxes(axs[3, 2])


# selects plots in main
def plotter(corr_dict, labels, dataset_trained, dataset_extracted, single_plot, multi_plot, rdm_plot):
    if single_plot:
        # single plotting
        for model, correlation in corr_dict.items():
            plot_rdm(correlation)
            plt.title(join("1-CorrelationMatrix on 500 inputs of ", model), weight='semibold')
            plt.show()

    if multi_plot:
        # multi plot multiple Corr_matrices
        print('> Multi-plot all models')
        multi_plot_rdm(corr_dict, labels)
        # adding 'title' doesn't work, since takes last ax element -- how to plot into fig space without call figure?
        # plt.text(0.9, 0.1, f"Input RDMs \n pre: {dataset_trained} \n extracted: {dataset_extracted}", weight='semibold',
        #           fontsize=12, transform=plt.gca().transAxes)
        plt.show()

    if rdm_plot:
        # RDM
        print('> Calc and plot RDM: of all models layer4 correlations')
        # corr_dict_layer4 activations into RDM calculation, keys for plotting
        rdm = calc_rdm(distance.euclidean, list(corr_dict.values()))  ## with euclid dist
        plot_rdm(rdm, list(corr_dict.keys()))
        plt.title("Model RDM - pre: " + dataset_trained + " / extracted: " + dataset_extracted, weight='semibold',
                  fontsize=20)
        plt.show()

    # return rdm


def load_calc_corr(dataset_trained, dataset_extracted, sorted):
    # load
    root_path = os.getcwd()
    models_dir = join(root_path, 'models', dataset_trained)
    # TO-DO: for i in range(1, 11):  # for 10 seed folders
    models_dir = join(models_dir, 'models_1')
    load_extracted = join(models_dir, dataset_extracted + '_extracted.pt')
    models = torch.load(load_extracted)
    print('loaded - ', load_extracted)
    print('nr of models: ', len(models))
    # print('loaded model list: ', [name for name, model in natsorted(models.items())])

    # get labels once (since constant) for plotting
    labels = list(models.values())[0]['labels']  # tensor, same for all models - use later for plots
    print('labels and count: ', np.unique(np.array(labels), return_counts=True))

    # calculate or load Correlation_Dictionary for one layer
    path = join(models_dir, dataset_extracted + '_corr_dict_layer4.pik')
    if sorted==True: path = join(models_dir, dataset_extracted + '_sorted_corr_dict_layer4.pik')

    if not os.path.exists(path):
        # on out of encoder so layer=4
        a = time.time()
        corr_dict_layer4 = calculate_activations_correlations(models, layer=4, sorted=sorted)
        print('time for corr_dict calculation: ', time.time() - a)
        # save it in models folder
        with open(str(path), 'wb') as f:
            dill.dump(corr_dict_layer4, f)
    else:
        print('>> already calculated >> load cor_dict')
        with open(str(path), 'rb') as f:
            corr_dict_layer4 = dill.load(f)

    if sorted: labels = np.sort(np.array(labels))
    return corr_dict_layer4, labels


def main(dataset_trained, dataset_extracted, sorted):
    # load models and calc/load correlations
    corr_dict_layer4, labels = load_calc_corr(dataset_trained, dataset_extracted, sorted)

    # plots on one dataset
    plotter(corr_dict_layer4, labels, dataset_trained, dataset_extracted,
            single_plot=False, multi_plot=True, rdm_plot=True)

    # Model RDM of source and target (trained and extracted dataset)
    # if dataset_trained != dataset_extracted:
    #     path = join(models_dir, dataset_extracted + '_corr_dict_layer4.pik')
    #     print('>> already calculated >> load cor_dict')
    #     with open(str(path), 'rb') as f:
    #         corr_dict_layer4 = dill.load(f)

    return corr_dict_layer4



if __name__ == '__main__':

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Devise used = cuda on ", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Devise used = ", device)

    dataset_trained = 'mnist' # = dataset_extracted (both source dataset)
    corr_dict_source = main(dataset_trained, dataset_trained, sorted=True)

    dataset_extracted = 'fashionmnist'
    corr_dict_target = main(dataset_trained, dataset_extracted, sorted=True)  # on target dataset

    # dataset_trained =
    # dataset_extracted =
    # main(dataset_trained, dataset_extracted, sorted=False)

    # calc model RDM: version 1 on corr_matrices
    # overlapping dict keys (model names) - add dataset names
    corr_dict_source = {f'{k}@{dataset_trained}': v for k, v in corr_dict_source.items()}
    corr_dict_target = {f'{k}@{dataset_extracted}': v for k, v in corr_dict_target.items()}
    corr_dict_total = {}
    corr_dict_total.update(corr_dict_source)
    corr_dict_total.update(corr_dict_target)
    print(len(corr_dict_total))

    rdm = calc_rdm(distance.euclidean, list(corr_dict_total.values()))  ## with euclid dist
    plot_rdm(rdm, list(corr_dict_total.keys()))
    plt.title(f"Model RDM - pre: {dataset_trained} / extracted:{dataset_extracted}, {dataset_trained}",
              weight='semibold', fontsize=20)
    plt.show()

    # version 2: compare already calculated RDM values




# take extracted activations of 500 source sampels -- create RDM
# input 500 TARGET samples (classifier maybe doesn't make sense) extract activations -- create RDM
# -> calc correlation between these two RDMs = similarity of activations representation of different data in same NN
# ---> therefore predicts post-ft Acc since similar representations will do similarly good?

## from paper
# take from encoder output layer (so before fc layers)
# take 500 samples forwardpass activations on that layer - calculate RDM (pair-wise correlation)
# correlate these RDMs of different tasks -- with similarity score based on !!spearman corr!! [not euclid]

# What is the insight of that? Low trained models corr with other low trained, and high with high?
# How does that help identify the one best in Transferability? Choose the one with highest correlation to
# a RDM of the target data? Since one wants to TL the encoder part, it's the right architecture and the
# goal is that it produces good representation on target data once TL

# -->> GOAL RSA: Do forward-pass (500 img) on the pre-trained model once with source data and once with target
# -->>> Ideally the model with highest TL shows highest correlation of RDMS
# -->>>> So, without FT every pre-trained model, we know which one will have highest post-ft Acc? (Hypothisis to check)


