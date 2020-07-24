import numpy as np
import torch


def sum_squared(layer, labels):
    # convert to numpy if tensor
    if torch.is_tensor(layer):
        layer = torch.Tensor.numpy(layer)
    if torch.is_tensor(labels):
        labels = torch.Tensor.numpy(labels)

    # SS within clusters
    clusters = []
    cluster_means = []
    ss = []
    for label in np.unique(labels):
        clusters.append(layer[labels == label])
        cluster_means.append(np.mean(clusters[-1], axis=0))
        ss.append(np.sum((clusters[-1] - np.array([cluster_means[-1]])) ** 2))

    # TSS
    layer_mean = np.mean(layer, axis=0)
    tss = np.sum((layer - np.array([layer_mean]))**2)

    return tss, np.sum(ss)
    # then look at ssw/tss: should be as small as possible
    # (= small ss within clusters, samples tight around cluster mean, & tss large, diff. labels get separated far apart)


if __name__ == '__main__':

    layer = np.array([[1, 0.8], [1, 2], [3, 4], [4, 5], [4, 2], [7, 7], [5, 5]])
    labels = np.array([1, 1, 2, 2, 2, 3, 3])
    tss, ss = sum_squared(layer, labels)
    print('TSS: ', tss, ' SS: ', ss, ' ss/tss: ', ss / tss)

    # ex with tensor input
    layer = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = np.array([1, 2, 2])
    tss, ss = sum_squared(layer, labels)
    print(tss, ss)

    # for extracted activations, like in analyze.py
    path = '/mnt/antares_raid/home/bemmerl/thesis/data/mnist2class/models_2/_extracted.pt'
    models = torch.load(path)
    ss = []
    for name, model in models.items():
        # print(name)
        layers = model['layers']  # input + 6 model output layers
        labels = model['labels']
    for layer in layers:
        tss, ssw = sum_squared(layer, labels)
        ss.append(ssw / tss if tss != 0 else 0)
    print(ss)
