import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import dill
import time
import math
import os
from os.path import join
import scipy.stats
from scipy.spatial import distance
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import manhattan_distances
from tqdm import tqdm
from natsort import natsorted


def correlationd_matrix(activations):
    n = len(activations)
    correlationd = np.zeros((n, n))
    for i in tqdm(range(n)):
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
            inputs = inputs[np.array(labels).argsort(), :]

        corr_distances_dict[name] = correlationd_matrix(inputs)  ## corr_dist_dict = corr_matrix

    return corr_distances_dict


# alternative RDM corr. dist - used in paper (but euclid. better for NN, since preserves magnitudes
def spearman_dist(a, b):
    return 1 - spearmanr(a, b)[0]


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


def calc_compare_rdm(dist_fun, corr_dist_dict_source, corr_dist_dict_target):
    t_rdm = time.time()
    n = len(corr_dist_dict_source)
    print(len(corr_dist_dict_source) == len(corr_dist_dict_target))
    compare_rdm_diagonal = np.zeros(n)
    for i in range(n):
        compare_rdm_diagonal[i] = dist_between_corr_matrices(dist_fun,
                                                             corr_dist_dict_source[i],
                                                             corr_dist_dict_target[i])
    print(f'>> Calculate {n} RDM diagonal elements' + '(in {:.2f} s)'.format(time.time()-t_rdm))

    return compare_rdm_diagonal


def plot_rdm_compare(compare_rdm_euclid, compare_rdm_spearman, model_names, acc):
    fig, ax1 = plt.subplots(figsize=(8, 8), dpi=150)
    ax1.tick_params(axis='x', labelsize=8, labelrotation=60)
    ax1.set_xlabel('Models')

    ax1.set_ylabel('euclidean distance', color='r')
    ax1.plot(model_names, compare_rdm_euclid, label='euclidean', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('spearman distance', color='b')
    ax2.plot(model_names, compare_rdm_spearman, label='spearman', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    ax3 = ax1.twinx()
    # Make some space on the right side for the extra y-axis.
    fig.subplots_adjust(right=0.65)
    # Move the last y-axis spine over to the right by 20% of the width of the axes
    ax3.spines['right'].set_position(('axes', 1.2))
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)

    ax3.set_ylabel('Accuracy', color='g')
    ax3.plot(model_names, acc, marker='o', linestyle='none', label='Acc', color='g')
    ax3.tick_params(axis='y', labelcolor='g')

    # fig.legend()
    fig.tight_layout()


def plot_rdm(rdm, model_names):
    fig = plt.figure(figsize=(16, 14), dpi=150)
    ax = sns.heatmap(rdm, cmap='rainbow', annot=True, fmt='6.3g', xticklabels=model_names, yticklabels=model_names,
                         vmin=0, vmax=210)
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')


def multi_plot_rdm(corr_dict, labels):
    n = math.sqrt(len(corr_dict))
    fig, axs = plt.subplots(math.ceil(n), math.floor(n), figsize=(16, 21), sharex='col', sharey='row', dpi=250)
    fig.subplots_adjust(hspace=.5, wspace=.001)

    for ax, val in tqdm(zip(axs.ravel(), corr_dict.items())):
        sns.set(font_scale=0.8)
        sns.heatmap(val[1], cmap='rainbow', ax=ax, cbar=True, xticklabels=labels, yticklabels=labels)
        ax.set_title(val[0], weight='semibold')

        ax.tick_params(axis='both', labelsize=3, width=0.3, length=1)
        n = 10  # Keeps every nth label
        [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 0]
    fig.delaxes(axs[3, 2])
    plt.show()


def multi_plot_histo(corr_dict, labels):
    # histogram of corr_value distribution
    n = math.sqrt(len(corr_dict))
    fig2, axs2 = plt.subplots(math.ceil(n), math.floor(n), figsize=(16, 21), sharex='col', sharey='row', dpi=150)
    fig2.subplots_adjust(hspace=.5, wspace=.001)

    for ax2, val2 in tqdm(zip(axs2.ravel(), corr_dict.items())):
        sns.distplot(val2[1], ax=ax2)
        ax2.set_title(val2[0], weight='semibold')
        total = int(np.sum(val2[1]))
        ax2.text(0.85, 0.05, join('sum: ' + str(total) + ' (' + str(int(total/(500*500*2)*100)) + '% of total)'))
    fig2.delaxes(axs2[3, 2])
    plt.show()


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
        print('> Multi-plot all models and show distribution histogram')
        multi_plot_rdm(corr_dict, labels)
        multi_plot_histo(corr_dict, labels)

    if rdm_plot:
        print('> Calc and plot RDM: of all models layer4 correlations')
        # corr_dict_layer4 activations into RDM calculation, keys for plotting
        rdm = calc_rdm(distance.euclidean, list(corr_dict.values()))  ## with euclid dist
        # rdm = calc_rdm(spearman_dist, list(corr_dict.values()))
        plot_rdm(rdm, list(corr_dict.keys()))
        plt.title("Model RDM - pre: " + dataset_trained + " / extracted: " + dataset_extracted, weight='semibold',
                  fontsize=20)
        plt.show()


def load_calc_corr(dataset_trained, dataset_extracted, sorted, seed=1):
    # load
    root_path = os.getcwd()
    models_dir = join(root_path, 'models', dataset_trained, 'models_' + str(seed))

    load_extracted = join(models_dir, dataset_extracted + '_extracted.pt')
    models = torch.load(load_extracted)
    print(f'loaded {len(models)} models from - {load_extracted}')

    # get labels once (since constant) for plotting
    labels = list(models.values())[0]['labels']  # tensor, same for all models - use later for plots
    print('labels and count: ', np.unique(np.array(labels), return_counts=True))  # check if balanced data

    # calculate or load Correlation_Dictionary for one layer
    path = join(models_dir, dataset_extracted + '_corr_dict_layer4.pik')
    if sorted==True: path = join(models_dir, dataset_extracted + '_sorted_corr_dict_layer4.pik')

    if not os.path.exists(path):
        # on out of encoder, so layer=4
        a = time.time()
        corr_dict_layer4 = calculate_activations_correlations(models, layer=4, sorted=sorted)
        print('time for corr_dict calculation: ', time.time() - a)
        # save it in models folder
        with open(str(path), 'wb') as f:
            dill.dump(corr_dict_layer4, f)
        print(f'{path} saved.')
    else:
        print('>> already calculated >> load cor_dict')
        with open(str(path), 'rb') as f:
            corr_dict_layer4 = dill.load(f)

    if sorted: labels = np.sort(np.array(labels))
    return corr_dict_layer4, labels


def main(dataset_trained, dataset_extracted, sorted, seed=1):
    # load models and calc/load correlations
    corr_dict_layer4, labels = load_calc_corr(dataset_trained, dataset_extracted, sorted, seed=seed)

    # plots on one dataset
    plotter(corr_dict_layer4, labels, dataset_trained, dataset_extracted,
            single_plot=False, multi_plot=False, rdm_plot=False)

    return corr_dict_layer4


# function to get called in analyze.py to return compare_rdm values for all seeds
def get_rdm_metric(source, target):
    total_compare_rdm = []
    for seed in range(1, 11):
        source_corr_dict_layer4, _ = load_calc_corr(source, source, sorted=True, seed=seed)
        target_corr_dict_layer4, _ = load_calc_corr(source, target, sorted=True, seed=seed)

        compare_rdm = calc_compare_rdm(distance.euclidean,
                                       list(source_corr_dict_layer4.values()),
                                       list(target_corr_dict_layer4.values()))
        total_compare_rdm.append(compare_rdm)

    return np.concatenate(total_compare_rdm).ravel().tolist()


if __name__ == '__main__':
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Devise used = cuda on ", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Devise used = ", device)

    # set source(trained) and target(extracted) datasets
    dataset_trained = 'mnist'
    corr_dict_source = main(dataset_trained, dataset_trained, sorted=True, seed=1)  # only plot for seed 1

    dataset_extracted = 'fashionmnist'
    corr_dict_target = main(dataset_trained, dataset_extracted, sorted=True, seed=1)


    # calculate only diagonal of model RDM (=corr/dist of same model for source and target data)
    compare_rdm_euclid = calc_compare_rdm(distance.euclidean, list(corr_dict_source.values()), list(corr_dict_target.values()))
    compare_rdm_spearman = calc_compare_rdm(spearman_dist, list(corr_dict_source.values()), list(corr_dict_target.values()))

    # load Acc to add to plot
    df = pd.read_pickle(join(os.getcwd(), 'models', dataset_trained, 'df_pre_' + dataset_trained))
    acc = df[df['seed'] == 1]['pre_test_acc']

    plot_rdm_compare(compare_rdm_euclid, compare_rdm_spearman, list(corr_dict_source.keys()), acc)
    plt.title(f"Compare RDM values - for all models pre:{dataset_trained} on {dataset_extracted} & {dataset_trained}",
              weight='semibold', fontsize=11.5)
    plt.show()





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


