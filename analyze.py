import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from IDNN.intrinsic_dimension import estimate, block_analysis
import sum_of_squares


def computeID(r, number_resampling=50, fraction=0.9, distance_metric='euclidean'):
    ID = []
    n = int(np.round(r.shape[0] * fraction))
    dist = squareform(pdist(r, distance_metric))
    for i in range(number_resampling):
        dist_s = dist
        perm = np.random.permutation(dist.shape[0])[0:n]
        dist_s = dist_s[perm, :]
        dist_s = dist_s[:, perm]
        ID.append(estimate(dist_s, verbose=True)[2])
    mean = np.mean(ID)
    error = np.std(ID)
    return mean, error


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--file', type=str, required=True)
    # args = parser.parse_args()
    # models = torch.load(args.file)

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Devise used = cuda on ", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Devise used = ", device)

    models = torch.load('/mnt/antares_raid/home/bemmerl/thesis/data/mnist/extracted.pt', map_location='cpu')
    print(len(models))
    all_ids = []
    all_ss = []
    for name, model in models.items():
        print(name)
        layers = model['layers']
        labels = model['labels']
        ids = []
        ss = []
        for layer in layers:
            id, error = computeID(layer)
            ids.append(id)
            tss, ss_mean = sum_of_squares.sum_squared(layer, labels)
            ss.append(ss_mean / tss if tss != 0 else 0)

        all_ids.append(ids)
        all_ss.append(ss)

    print(all_ss, all_ids)

    # plot ids
    plt.figure()
    for name, ids in zip(models.keys(), all_ids):
        print(name, ids)
        plt.plot(range(len(ids)), ids, label=name)
    plt.legend()
    plt.show()

    for name, ss in zip(models.keys(), all_ss):
        plt.plot(range(len(ss)), ss, label=name)
    plt.legend()
    plt.show()
