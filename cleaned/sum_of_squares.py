import numpy as np
from scipy.spatial.distance import cdist
import torch
from collections import Counter


def sum_squared(layer, labels):
    if torch.is_tensor(layer):
        layer = torch.Tensor.numpy(layer)
        #print('tensor passed - converted to numpy')
    if torch.is_tensor(labels):
        labels = torch.Tensor.numpy(labels)

    # SS within clusters
    clusters = []
    cluster_means = []
    ss = []
    for label in np.unique(labels):
        #print(label)
        clusters.append(layer[labels == label])
        cluster_means.append(np.mean(clusters[-1], axis=0))
        ss.append(np.sum(cdist(clusters[-1], np.array([cluster_means[-1]]))))
        #print(ss)

    # TSS
    layer_mean = np.mean(layer, axis=0)
    tss = np.sum(cdist(layer, np.array([layer_mean])))

    # distances from every cluster mean to the total mean
    t_mean_dist = cdist(np.array(cluster_means), np.array([layer_mean]))
    # add dist to mean for nr of pts to SSW cluster
    ss_total = np.sum(ss)
    #print(Counter(labels))
    for key in Counter(labels).keys():
        ss_total += Counter(labels)[key] * t_mean_dist[key - 1].item()
    return tss, np.mean(ss), ss_total


if __name__ == '__main__':
    layer = np.array([[1, 0.8], [1, 2], [3, 4], [4, 5], [4, 2], [7, 7], [5, 5]])
    labels = np.array([1, 1, 2, 2, 2, 3, 3])
    tss, ss_mean, ss_total = sum_squared(layer, labels)

    print('TSS: ', tss)
    print('SSW: ', ss_total)
    print('-> SSW/TSS: ', ss_total / tss)
    print('compare mean(ss)/tss: ', ss_mean / tss)

    # ex with tensor input
    layer = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = np.array([1, 2, 2])
    tss, ss_mean, ss_total = sum_squared(layer, labels)
    print(tss, ss_mean, ss_total)


    path = '/mnt/antares_raid/home/bemmerl/thesis/data/mnist2class/models_2/_extracted.pt'
    models = torch.load(path)
    all_ids = []
    all_ss = []
    ss = []
    compare= []
    for name, model in models.items():
        print(name)
        layers = model['layers']  # input + 6 model output layers
        labels = model['labels']
    for layer in layers:
        #print(layer, labels)
        tss, ss_mean, ssw = sum_squared(layer, labels)
        #print(ssw, tss)
        ss.append(ssw / tss if tss != 0 else 0)
        compare.append(ss_mean / tss if tss != 0 else 0)
    print(ss)
    print(compare)
