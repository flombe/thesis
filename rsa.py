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




def calculate_activations_correlations(files, path, num_samples=1000, show_progress=True, activations_dict={},
                                       corr_distances_dict={}, confusion_dict={}, calc_confusion=False):

    # dataloader_valid = torch.utils.data.DataLoader(MNIST_valid, batch_size=num_samples, shuffle=False)
    # inputs, labels = next(iter(dataloader_valid))
    # n = len(labels)

    # def calculate_confusion(model, inputs, labels):
    #     confusion_matrix = np.zeros((10, 10))
    #
    #     with torch.no_grad():
    #         model.eval()
    #
    #         inputs, labels = inputs.cuda(), labels.cuda()
    #
    #         out = model(inputs)[0]
    #
    #         for true_label in range(10):
    #
    #             true_index = (labels == true_label)
    #
    #             if (true_index.sum() != 0):
    #                 (torch.max(out, 1)[1])
    #                 for output in torch.max(out[true_index], 1)[1]:
    #                     for estimate in range(10):
    #                         confusion_matrix[true_label, estimate] += (output == estimate).sum()
    #             else:
    #                 pass
    #
    #     return (confusion_matrix)
    #
    # def activations_list(model):
    #     activations = []
    #     with torch.no_grad():
    #         model.eval()
    #         for i in range(n):
    #             if model_type == ConvNet:
    #                 npt = inputs[i].cuda().view(1, 1, 28, 28)
    #             else:
    #                 npt = inputs[i].cuda()
    #             activations.append(model(npt)[1].cpu().numpy().squeeze())
    #     return (activations)

    def correlationd_matrix(list_of_activations):
        correlationd = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                correlationd[i, j] = correlationd[j, i] = 1 - scipy.stats.pearsonr(list_of_activations[i],
                                                                                   list_of_activations[j])[0]
        return (correlationd)

    for i in range(len(files, )):
        if show_progress:
            print(i)
        model_name = files[i]
        if path[i] == 0:
            real_path = 'gdrive/My Drive/NI Project - RSA/mnist_saves/'
            model_type = feedforward
            inputs, labels = torch.autograd.Variable(inputs).view((-1, 28 * 28)).float(), torch.autograd.Variable(
                labels).long()
        else:
            real_path = 'gdrive/My Drive/NI Project - RSA/mnist_saves_conv/'
            model_type = ConvNet

        model_dict = torch.load(f'{real_path}{model_name}')
        layers = model_dict['layers']
        model = model_type(*layers).cuda()

        model.load_state_dict(model_dict['state_dict'])

        activations_dict[i] = activations_list(model)
        corr_distances_dict[i] = correlationd_matrix(activations_dict[i])
        if calc_confusion:
            confusion_dict[i] = calculate_confusion(model, inputs, labels)

    return activations_dict, corr_distances_dict, confusion_dict







def dist_between_corr_matrices(dist_fun, corr_matrix_1, corr_matrix_2):
    triu_model_1 = corr_matrix_1[np.triu_indices(corr_matrix_1.shape[0], k=1)]
    triu_model_2 = corr_matrix_2[np.triu_indices(corr_matrix_2.shape[0], k=1)]

    return dist_fun(triu_model_1, triu_model_2)


def calc_rdm(dist_fun, corr_distances_dict,  files):
    n = len(files)
    rdm = np.zeros((n, n))
    for i in range(n):
        if i % 10 == 0:
            print(i)
        for j in range(i, n):
            rdm[i, j] = rdm[j, i] = dist_between_corr_matrices(dist_fun,
                                                               corr_distances_dict[i],
                                                               corr_distances_dict[j])
    return rdm


def visualise_rdm(rdm):
    plt.figure(figsize=(14 ,14))
    ax = seaborn.heatmap(rdm, cmap='rainbow')  # , xticklabels = np.around(acc_df['acc'].astype('float') * 100), yticklabels = np.around(acc_df['acc'].astype('float') * 100), cmap='rainbow')
#   n = 15  # Keeps every nth label
#   [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
#   [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 0]



if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Devise used = cuda on ", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Devise used = ", device)


    #for i in range(1, 11):  # for 10 seed folders
    path = join('/mnt/antares_raid/home/bemmerl/thesis/data/mnist2class/models_1')
    models = torch.load(join(path, '_extracted.pt'))

    print('nr of models: ', len(models))

    for name, model in tqdm(models.items()):
        print(name)
        layers = model['layers']  # input + 6 model output layers
        labels = model['labels']  # 20 labels to much

    # is last element of for loop = model_pre_mnist2_0batch100
    print(len(layers))
    print(len(labels))

    ## first test rdm of 10 first samples on last layer of one model
    select_layer = layers[5][0:10, :]  # second to last layer activations - of 100 samples
    print(select_layer.shape)
    select_labels = labels[0:10]
    print(select_labels)

    inputs = select_layer
    labels = select_labels
    n = len(labels)

    print(inputs)
    print(type(inputs))
    print(type(inputs.cpu().numpy().squeeze()))
    inputs = (inputs.cpu().numpy().squeeze())
    print(inputs)

    corr_distances_dict = {}
    def correlationd_matrix(list_of_activations):
        correlationd = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                print(i, j)
                correlationd[i, j] = correlationd[j, i] = 1 - scipy.stats.pearsonr(list_of_activations[i],
                                                                                   list_of_activations[j])[0] #[0] for pearson coeff
        return correlationd


    # for i in range(len(inputs)):
    #     print(i)
    #     print(inputs[i])
    #     corr_distances_dict[i] = correlationd_matrix(inputs[i])

    corr_distances_dict = correlationd_matrix(inputs)
    visualise_rdm(corr_distances_dict)
    print(corr_distances_dict)
    print(corr_distances_dict.shape)
    print(corr_distances_dict[0])
    print(np.triu_indices(corr_distances_dict.shape[0], k=1))
    triu = corr_distances_dict[np.triu_indices(corr_distances_dict.shape[0], k=1)]
    triu2 = corr_distances_dict[np.triu_indices(corr_distances_dict.shape[1], k=1)]
    print(triu)

    plt.figure(figsize=(14, 14))
    ax = seaborn.heatmap(corr_distances_dict, cmap='rainbow')
    plt.show()

    #rdm = calc_rdm(distance.euclidean, corr_distances_dict, list(inputs))
    #visualise_rdm(rdm)
