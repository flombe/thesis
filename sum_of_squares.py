import numpy as np
from scipy.spatial.distance import cdist
import torch


def sum_squared(layer, labels):
    if torch.is_tensor(layer):
        layer = torch.Tensor.numpy(layer)
        #print('tensor passed - converted to numpy')

    clusters = []
    cluster_means = []
    ss = []

    for label in np.unique(labels):
        clusters.append(layer[labels == label])
        cluster_means.append(np.mean(clusters[-1], axis=0))   ## convert or use torch.mean
        ss.append(np.sum(cdist(clusters[-1], np.array([cluster_means[-1]]))))

    layer_mean = np.mean(layer, axis=0)
    tss = np.sum(cdist(layer, np.array([layer_mean])))
    return tss, np.mean(ss)


if __name__ == '__main__':
    layer = np.array([[1, 2], [3, 4], [5, 6]])
    labels = np.array([1, 2, 2])
    tss, ss_mean = sum_squared(layer, labels)
    print(ss_mean, tss)

    layer = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = np.array([1, 2, 2])
    tss, ss_mean = sum_squared(layer, labels)
    print(ss_mean, tss)
