import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import os
from os.path import join
import pandas as pd
import json

from IDNN.intrinsic_dimension import estimate, block_analysis
import sum_of_squares
import rsa

# compute ID, SS (and RSA) on extracted activations

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


if __name__ == '__main__':
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Devise used = cuda on ", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Devise used = ", device)

    trained_dataset = 'mnist'  # trained on
    extract_dataset = 'mnist'  # extracted on
    root_dir = os.getcwd()
    models_dir = join(root_dir, 'models', trained_dataset)

    for seed in range(1, 11):  # for 10 seed folders
        print(f'>>Seed {seed} / pre-trained on {trained_dataset}, extracted on {extract_dataset}<<<')
        path = join(models_dir, 'models_' + str(seed))
        extract = torch.load(join(path, extract_dataset + '_extracted.pt'))  # map_location='cpu'

        all_ids = []
        all_ss = []
        for name, model in tqdm(extract.items()):
            print(name)
            layers = model['layers']  # input + 6 model output layers
            labels = model['labels']

            ids = []
            ss = []
            for layer in layers:
                id, error = computeID(layer)
                ids.append(id)
                tss, ssw = sum_of_squares.sum_squared(layer, labels)
                ss.append(ssw / tss if tss != 0 else 0)

            all_ids.append(ids)
            all_ss.append(ss)


        ## df



        results = zip(extract.keys(), all_ss, all_ids)
        print(list(results))
        order = ['0batch1', '0batch3', '0batch10', '0batch30', '0batch100', '0batch300', '1', '3', '10', '30', '100']
        new_list = list()
        # for j in range(len(order)):
        #     for i in range(len(order)):
        #         for line in zip(models.keys(), all_ss, all_ids):
        #             if line[0].endswith(join('_' + order[j] + '_' + order[i] + '.pt')):
        #                 new_list.append(line)
        #                 break

        for i in range(len(order)):
            for line in zip(extract.keys(), all_ss, all_ids):
                if line[0].endswith(join('_' + order[i] + '.pt')):
                    new_list.append(line)
                    break

        print('---- 0batch1 ?: ', new_list[0:11])
        print('---- 0batch3 ?: ', new_list[11:22])
        print('---- 0batch10 ?: ', new_list[22:33])
        break
        analytics = {
            'model_pre_mnist2': new_list,
        }

        # analytics = {
        #     'model_ft_mnist2_mnist_0batch1' : new_list[0:11],
        #     'model_ft_mnist2_mnist_0batch3': new_list[11:22],
        #     'model_ft_mnist2_mnist_0batch10': new_list[22:33],
        #     'model_ft_mnist2_mnist_0batch30': new_list[33:44],
        #     'model_ft_mnist2_mnist_0batch100': new_list[44:55],
        #     'model_ft_mnist2_mnist_0batch300': new_list[55:66],
        #     'model_ft_mnist2_mnist_1': new_list[66:77],
        #     'model_ft_mnist2_mnist_3': new_list[77:88],
        #     'model_ft_mnist2_mnist_10': new_list[88:99],
        #     'model_ft_mnist2_mnist_30': new_list[99:110],
        #     'model_ft_mnist2_mnist_100': new_list[110:121]
        #     }
        with open(join(path, 'ss_id.json'), 'w+') as f:
            json.dump(analytics, f)

# output dict keys = pretrained model finetuned and value = ('model checkpoint', [ss for 7 layers], [id for 7 layers])
