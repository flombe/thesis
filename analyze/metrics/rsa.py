import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dill
import time
import math
import os
from os.path import join
import scipy.stats
from scipy.spatial import distance
from scipy.stats import spearmanr
from tqdm import tqdm
from natsort import natsorted
from skimage.util import view_as_blocks
import sklearn.manifold


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
        layers = model['layers']
        inputs = layers[layer]

        if sorted==True:
            inputs = inputs[np.array(labels).argsort(), :]

        corr_distances_dict[name] = correlationd_matrix(inputs)

    return corr_distances_dict

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

def plot_rdm(rdm, model_names):
    fig = plt.figure(figsize=(16, 14), dpi=150)
    ax = sns.heatmap(rdm, cmap='rainbow', annot=True, fmt='6.3g', xticklabels=model_names, yticklabels=model_names,
                         vmin=0, vmax=210)
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')

def multi_plot_rdm(corr_dict, labels):
    n = math.sqrt(len(corr_dict))
    fig, axs = plt.subplots(math.ceil(n), math.floor(n), figsize=(16, 21), sharex='col', sharey='row', dpi=250)
    fig.subplots_adjust(hspace=.5, wspace=.001)

    for ax, val in tqdm(zip(fig.axes, corr_dict.items())):
        sns.set(font_scale=0.8)

        print(val[1])

        sns.heatmap(val[1], cmap='rainbow', ax=ax, cbar=True, xticklabels=labels, yticklabels=labels, vmax=1.5)
        ax.set_title(val[0], weight='semibold')

        ax.tick_params(axis='both', labelsize=3, width=0.3, length=1)
        n = 10  # Keeps every nth label
        [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
        [l.set_visible(False) for (i, l) in enumerate(ax.yaxis.get_ticklabels()) if i % n != 0]
    plt.show()

def multi_plot_histo(corr_dict, labels):
    # histogram of corr_value distribution
    n = math.sqrt(len(corr_dict))
    fig2, axs2 = plt.subplots(math.ceil(n), math.floor(n), figsize=(8, 10), sharex='col', sharey='row', dpi=150)
    fig2.subplots_adjust(hspace=.5, wspace=.001)

    if dataset_extracted == 'custom3D':
        layer_names = ['in', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc1', 'fc2', 'out']
        block_size = 30
        nr_lables = 40
    else:
        layer_names = ['in', 'conv1', 'pool1', 'conv2', 'pool2', 'fc1', 'output']
        block_size = 50
        nr_lables = 10

    for ax2, val2 in tqdm(zip(fig2.axes, corr_dict.items())):
        block_view = view_as_blocks(val2[1], block_shape=(block_size, block_size))

        val_diag = np.array([])
        val_blockdiag = np.array([])
        val_nondiag = np.array([])

        for i in range(nr_lables):
            for j in range(nr_lables):
                if i == j:  # diagonal block
                    B = block_view[i, j]
                    # print('Block matrix: ', B.shape)
                    val_diag = np.append(val_diag, B.diagonal())
                    # print('Diagonal: ', B.diagonal().shape)
                    B = B[~np.eye(B.shape[0], dtype=bool)].reshape(B.shape[0], -1)  # drop diagonal elements
                    # print('Block-diagonal without diagonal: ', B.shape)
                    val_blockdiag = np.append(val_blockdiag, B)
                    # print(val_blockdiag.shape)
                else:
                    val_nondiag = np.append(val_nondiag, block_view[i, j])
                    # print(val_nondiag.shape)

        print(f'- end shape: diag {val_diag.shape} / block-diag {val_blockdiag.shape} / non-diag {val_nondiag.shape}')
        sns.distplot(val_blockdiag, ax=ax2, label='Corr.Matrix block diagonal')
        sns.distplot(val_nondiag, ax=ax2, label='Corr.Matrix non-diagonal')
        sns.distplot(val_diag, bins=[-0.004, 0.004], ax=ax2, label='Corr.Matrix diagonal')

        # sns.distplot(val2[1], ax=ax2)
        ax2.set_title(val2[0], weight='semibold')
        ax2.set_ylim(0, 13)

        # add mean line to histogram
        kdeline_diag = ax2.lines[0]
        mean = val_blockdiag.mean()
        height = np.interp(mean, kdeline_diag.get_xdata(), kdeline_diag.get_ydata())
        ax2.vlines(mean, 0, height, color='blue', ls=':')

        kdeline = ax2.lines[1]
        mean = val_nondiag.mean()
        height = np.interp(mean, kdeline.get_xdata(), kdeline.get_ydata())
        ax2.vlines(mean, 0, height, color='crimson', ls=':')

    plt.legend(frameon=True, fancybox=True, facecolor='white')
    plt.show()

def MDS_plot(corr_dict, labels):

    rdm = calc_rdm(distance.euclidean, list(corr_dict.values()))
    model_names = list(corr_dict.keys())

    # MDS embedding
    embedding = sklearn.manifold.MDS(n_components=2, metric=True, dissimilarity='precomputed')
    embed_rdm = embedding.fit_transform(rdm)
    print(embed_rdm)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(embed_rdm[:, 0], embed_rdm[:, 1], cmap='rainbow', c=range(12), label=model_names)

    for i, txt in enumerate(model_names):
        ax.annotate(txt[16:-3], (embed_rdm[i, 0], embed_rdm[i, 1]), textcoords='offset points', xytext=(2, 2))

# selects plots in main
def plotter(corr_dict, labels, dataset_trained, dataset_extracted,
            single_plot, multi_plot, multi_plot_hist, mds_plot, rdm_plot):

    if single_plot:
        for model, correlation in corr_dict.items():
            plot_rdm(correlation, labels)
            plt.title(join("1-CorrelationMatrix on 500 inputs of ", model), weight='semibold')
            plt.show()

    if multi_plot:
        # multi plot multiple Corr_matrices
        print('> Multi-plot all models')
        multi_plot_rdm(corr_dict, labels)

    if multi_plot_hist:
        print('> Distribution histogram of all models')
        multi_plot_histo(corr_dict, labels)

    if mds_plot:
        MDS_plot(corr_dict, labels)
        plt.title("MDS of RDM - pre: " + dataset_trained + " / extracted: " + dataset_extracted, weight='semibold',
                  fontsize=12)
        plt.show()

    if rdm_plot:
        print('> Calc and plot RDM: of all models layer4 correlations')
        rdm = calc_rdm(distance.euclidean, list(corr_dict.values()))
        plot_rdm(rdm, list(corr_dict.keys()))
        plt.title("Model RDM - pre: " + dataset_trained + " / extracted: " + dataset_extracted, weight='semibold',
                  fontsize=20)
        plt.show()

def load_calc_corr(dataset_trained, dataset_extracted, sorted, seed, layer=4):
    # load
    root_path = os.getcwd()
    if dataset_extracted in ['custom3D', 'malaria', 'pets']:
        models_dir = join(root_path, '../../models', 'vgg16', dataset_trained)
        # if dataset_trained == 'random_init':
        #     models_dir = join(root_path, 'models', 'vgg16', dataset_trained, 'models_1')
    else:
        models_dir = join(root_path, '../../models', dataset_trained, 'models_' + str(seed))
        if not os.path.exists(models_dir):
            models_dir = join(root_path, '../../models', dataset_trained)

    load_extracted = join(models_dir, dataset_extracted + '_extracted.pt')
    models = torch.load(load_extracted)
    print(f'loaded {len(models)} models from - {load_extracted}')

    # get labels once (since constant) for plotting
    labels = list(models.values())[0]['labels']
    print('labels and count: ', np.unique(np.array(labels), return_counts=True))

    # calculate or load Correlation_Dictionary for one layer
    path = join(models_dir, dataset_extracted + f'_corr_dict_layer{layer}.pik')
    if sorted==True: path = join(models_dir, dataset_extracted + f'_sorted_corr_dict_layer{layer}.pik')


    if not os.path.exists(path):
        # on out of encoder, so layer=4, or otherwise specified
        a = time.time()
        corr_dict_layer = calculate_activations_correlations(models, layer=layer, sorted=sorted)
        print('time for corr_dict calculation: ', time.time() - a)
        # save it in models folder
        with open(str(path), 'wb') as f:
            dill.dump(corr_dict_layer, f)
        print(f'{path} saved.')
    else:
        print('>> already calculated >> load cor_dict')
        with open(str(path), 'rb') as f:
            corr_dict_layer = dill.load(f)

    if sorted: labels = np.sort(np.array(labels))
    return corr_dict_layer, labels

def main(dataset_trained, dataset_extracted, sorted, seed=1, layer=4):
    # load models and calc/load correlations
    corr_dict_layer4, labels = load_calc_corr(dataset_trained, dataset_extracted, sorted, seed=seed, layer=layer)

    # plots on one dataset
    plotter(corr_dict_layer4, labels, dataset_trained, dataset_extracted,
            single_plot=False,
            multi_plot=True,
            multi_plot_hist=True,
            mds_plot=False,
            rdm_plot=False)

    return corr_dict_layer4

# function to get called in analyze.py to return correlation delta on all layers for all models and seeds
def get_rdm_metric(source, target):
    total_deltas = []
    for seed in range(1, 11):
        layer_deltas = []

        for layer in range(7):
            corr_dict,_ = load_calc_corr(source, target, sorted, seed=seed, layer=layer)
            delta = []

            for model in corr_dict.items():
                block_view = view_as_blocks(model[1], block_shape=(50, 50))  # only for 10 classes x 50 samples
                val_diag = np.array([])
                val_blockdiag = np.array([])
                val_nondiag = np.array([])
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            B = block_view[i, j]
                            val_diag = np.append(val_diag, B.diagonal())
                            B = B[~np.eye(B.shape[0], dtype=bool)].reshape(B.shape[0], -1)  # drop diagonal elements
                            val_blockdiag = np.append(val_blockdiag, B)
                        else:
                            val_nondiag = np.append(val_nondiag, block_view[i, j])

                delta.append(abs(val_blockdiag.mean() - val_nondiag.mean()))  # take absolut value for metric

            layer_deltas.append(delta)  # list of lists [[12], ..7]

        for i in range(0, 12):  # nr of checkpts
            print([item[i] for item in layer_deltas])
            total_deltas.append([item[i] for item in layer_deltas])  # resort to [[7], ..12]
        print(len(total_deltas))

    print(len(total_deltas))  # 120
    print(total_deltas)
    return total_deltas

# for vgg16 architecture
def get_rdm_metric_vgg(source, target):
    if target == 'custom3D':
        block_size = 30
        class_count = 40
    elif target == 'malaria':
        block_size = 50
        class_count = 2
    elif target == 'pets':
        block_size = 30
        class_count = 37

    layer_deltas = []
    total_deltas = []

    if source in ['segnet', 'cifar10']: nr_layers = 6
    else: nr_layers = 9

    for layer in range(nr_layers):
        corr_dict, _ = load_calc_corr(source, target, sorted, seed=1, layer=layer)

        delta = []
        for model in corr_dict.items():  # corr_dict is sorted
            block_view = view_as_blocks(model[1], block_shape=(block_size, block_size))
            val_diag = np.array([])
            val_blockdiag = np.array([])
            val_nondiag = np.array([])
            for i in range(class_count):
                for j in range(class_count):
                    if i == j:
                        B = block_view[i, j]
                        val_diag = np.append(val_diag, B.diagonal())
                        B = B[~np.eye(B.shape[0], dtype=bool)].reshape(B.shape[0], -1)
                        val_blockdiag = np.append(val_blockdiag, B)
                    else:
                        val_nondiag = np.append(val_nondiag, block_view[i, j])
            delta.append(abs(val_blockdiag.mean() - val_nondiag.mean()))

        layer_deltas.append(delta)  # list of lists [[1], ..9]

    for i in range(0, 1):
        total_deltas.append([item[i] for item in layer_deltas])  # resort to [[9], ..1]

    return total_deltas[0]

def all_layer_plot(dataset_trained, dataset_extracted, sorted, seed=1):
    if dataset_extracted == 'custom3D':
        layer_names = ['in', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc1', 'fc2', 'out']
        block_size = 30
    else:
        layer_names = ['in', 'conv1', 'pool1', 'conv2', 'pool2', 'fc1', 'output']
        block_size = 50

    plt.figure(figsize=(10, 10))
    for layer in range(len(layer_names)):
        corr_dict,_ = load_calc_corr(dataset_trained, dataset_extracted, sorted, seed=seed, layer=layer)
        diag_mean = []
        nondiag_mean = []
        for model in corr_dict.items():
            block_view = view_as_blocks(model[1], block_shape=(block_size, block_size))
            val_diag = np.array([])
            val_nondiag = np.array([])
            for i in range(10):
                for j in range(10):
                    if i == j:
                        val_diag = np.append(val_diag, block_view[i, j])
                    else:
                        val_nondiag = np.append(val_nondiag, block_view[i, j])
            diag_mean.append(val_diag.mean())
            nondiag_mean.append(val_nondiag.mean())
        if layer_names[layer] == layer:
            p = plt.plot(range(len(corr_dict.items())), diag_mean, '--', label=f'{layer_names[layer]}_diag', alpha=0.2)
            plt.plot(range(len(corr_dict.items())), nondiag_mean, linestyle='dotted', label=f'{layer_names[layer]}_nondiag', c=p[0].get_color(), alpha=0.2)
            plt.plot(range(len(corr_dict.items())), [x1 - x2 for (x1, x2) in zip(nondiag_mean, diag_mean)], label=f'{layer_names[layer]}_delta', c=p[0].get_color())

    plt.xlabel('models')
    plt.ylabel('mean correlation')
    plt.title(f'RSA mean Corr. all layers - Diagonal Delta \n (pre: {dataset_trained}, on: {dataset_extracted})')
    plt.xlim(0, len(corr_dict.items())-1)
    plt.ylim(0, 1)
    plt.xticks(range(len(corr_dict.items())), list(corr_dict.keys()), rotation=70)
    plt.legend()
    plt.show()

def all_delta_plot(dataset_extracted, sorted, seed=1):
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 10))
    layer = 4

    if dataset_extracted == 'custom3D':
        layer_names = ['in', 'pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc1', 'fc2', 'out']
        block_size = 30
        nr_lables = 40
        checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '1', '3', '10', '30', '100']
        datasets = ['imagenet', 'places365', 'cars', 'vggface', 'segnet', 'cifar10', 'random_init', 'custom3D']
    else:
        layer_names = ['in', 'conv1', 'pool1', 'conv2', 'pool2', 'fc1', 'output']
        block_size = 50
        nr_lables = 10
        checkpts = ['0', '0_1', '0_3', '0_10', '0_30', '0_100', '0_300', '1', '3', '10', '30', '100']
        datasets = ['mnist_noise', 'mnist_noise_struct', 'mnist_split1', 'mnist', 'fashionmnist']


    for pre_set in datasets:
        corr_dict,_ = load_calc_corr(pre_set, dataset_extracted, sorted, seed=seed, layer=layer)
        diag_mean = []
        nondiag_mean = []
        for model in corr_dict.items():
            block_view = view_as_blocks(model[1], block_shape=(block_size, block_size))
            val_diag = np.array([])
            val_nondiag = np.array([])
            for i in range(nr_lables):
                for j in range(nr_lables):
                    if i == j:
                        val_diag = np.append(val_diag, block_view[i, j])
                    else:
                        val_nondiag = np.append(val_nondiag, block_view[i, j])
            diag_mean.append(val_diag.mean())
            nondiag_mean.append(val_nondiag.mean())

        if len(diag_mean)==11:
            p = plt.plot(range(12)[1:], diag_mean, '--', label=f'{pre_set}_diag', alpha=0.2)
            plt.plot(range(12)[1:], nondiag_mean, linestyle='dotted', label=f'{pre_set}_nondiag', c=p[0].get_color(), alpha=0.2)
            plt.plot(range(12)[1:], [x1 - x2 for (x1, x2) in zip(nondiag_mean, diag_mean)], label=f'{pre_set}_delta', c=p[0].get_color())
        else:
            p = plt.plot(range(12), diag_mean, '--', label=f'{pre_set}_diag', alpha=0.2)
            plt.plot(range(12), nondiag_mean, linestyle='dotted', label=f'{pre_set}_nondiag', c=p[0].get_color(), alpha=0.2)
            plt.plot(range(12), [x1 - x2 for (x1, x2) in zip(nondiag_mean, diag_mean)], label=f'{pre_set}_delta', c=p[0].get_color())

    plt.xlabel('models pre-train duration')
    plt.ylabel('mean correlation')
    plt.title(f'RSA mean Corr. Diagonal Delta (extracted: {dataset_extracted}, layer 4)', weight='semibold')
    plt.xlim(0, 11)
    plt.ylim(0, 1)
    plt.xticks(range(12), checkpts)
    plt.legend(frameon=True, fancybox=True, facecolor='white')
    plt.show()


if __name__ == '__main__':
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Devise used = cuda on ", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Devise used = ", device)

    # specify layer to analyze on
    layer = 2   # 5 for vgg

    # set source(trained) and target(extracted) datasets
    dataset_trained = 'random_init'  # vgg16/cifar10
    # corr_dict_source = main(dataset_trained, dataset_trained, sorted=True, seed=1, layer=layer)  # only plot for seed 1

    dataset_extracted = 'custom3D'
    corr_dict_target = main(dataset_trained, dataset_extracted, sorted=True, seed=1, layer=layer)

    # plot to compare corr means of all layers for all models
    all_layer_plot(dataset_trained, dataset_extracted, sorted=True, seed=1)

    all_delta_plot(dataset_extracted, sorted=True, seed=1)
