import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from os.path import join
import json

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
        ID.append(estimate(dist_s, verbose=False)[2])
    mean = np.mean(ID)
    error = np.std(ID)
    return mean, error


# sort models by training-logic for plotting
def model_sort(model_list):
    model_order = ['0batch1', '0batch3', '0batch10', '0batch30', '0batch100', '0batch300', '1', '3', '10', '30', '100']
    sorted_models = []
    for i in range(len(model_order)):
        for key in model_list:
            if key.endswith(join(model_order[i] + '.pt')):
                sorted_models.append(key)
    return sorted_models



if __name__ == '__main__':
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Devise used = cuda on ", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Devise used = ", device)

    for i in range(1,11): # for 10 seed folders
        path = join('/mnt/antares_raid/home/bemmerl/thesis/data/mnist/models_' + str(i))
        models = torch.load(join(path, '_extracted.pt')) #map_location='cpu'
        all_ids = []
        all_ss = []
        for name, model in tqdm(models.items()):
            print(name)
            layers = model['layers']  # input + 6 model output layers
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
        print(zip(models.keys(), all_ss, all_ids))

        results = zip(models.keys(), all_ss, all_ids)
        order = ['0batch1', '0batch3', '0batch10', '0batch30', '0batch100', '0batch300', '1', '3', '10', '30', '100']
        new_list = list()
        for j in range(len(order)):
            for i in range(len(order)):
                for line in zip(models.keys(), all_ss, all_ids):
                    if line[0].endswith(join(order[j] + '_' + order[i] + '.pt')):
                        new_list.append(line)
                        break

        print('---- 0batch1 ?: ', new_list[0:11])
        print('---- 0batch3 ?: ', new_list[11:22])
        print('---- 0batch10 ?: ', new_list[22:33])

        analytics = {
            'model_ft_mnist2_mnist_0batch1' : new_list[0:11],
            'model_ft_mnist2_mnist_0batch3': new_list[11:22],
            'model_ft_mnist2_mnist_0batch10': new_list[22:33],
            'model_ft_mnist2_mnist_0batch30': new_list[33:44],
            'model_ft_mnist2_mnist_0batch100': new_list[44:55],
            'model_ft_mnist2_mnist_0batch300': new_list[55:66],
            'model_ft_mnist2_mnist_1': new_list[66:77],
            'model_ft_mnist2_mnist_3': new_list[77:88],
            'model_ft_mnist2_mnist_10': new_list[88:99],
            'model_ft_mnist2_mnist_30': new_list[99:110],
            'model_ft_mnist2_mnist_100': new_list[110:121]
            }
        with open(join(path, 'ss_id.json'), 'w+') as f:
            json.dump(analytics, f)

# output dict keys = pretrained model finetuned and value = ('model checkpoint', [ss for 7 layers], [id for 7 layers])
