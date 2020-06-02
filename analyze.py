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
        path = join('/mnt/antares_raid/home/bemmerl/thesis/data/mnist2class/models_' + str(i))
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
        #print(all_ss, all_ids)

        results = zip(models.keys(), all_ss, all_ids)
        order = ['0batch1', '0batch3', '0batch10', '0batch30', '0batch100', '0batch300', '1', '3', '10', '30', '100']
        new_list = list()
        for i in range(len(order)):
            for line in zip(models.keys(), all_ss, all_ids):
                if line[0].endswith(join(order[i] + '.pt')):
                    new_list.append(line)
                    break

        analytics = {
            'model_pre_mnist2' : new_list
            }
        with open(join(path, 'ss_id.json'), 'w+') as f:
            json.dump(analytics, f)



    file = join('/mnt/antares_raid/home/bemmerl/thesis/data/mnist2class/models_1', 'ss_id.json')
    with open(file, 'r') as myfile:
        data = myfile.read()
    new_list = json.loads(data)['model_pre_mnist2']
    print(new_list)

    # plot ids
    plt.figure(figsize=(7, 6), dpi=100)
    for name, ss, ids in new_list:
        plt.plot(range(len(ids)), ids, label=name)
    plt.xlabel("Layers")
    plt.ylabel("Intrinsic Dimension")
    plt.legend()
    plt.show()

    for name, ss, ids in new_list:
        plt.plot(range(len(ss)), ss, label=name)
    plt.xlabel("Layers")
    plt.ylabel("SSW/TSS")
    plt.legend()
    plt.show()
