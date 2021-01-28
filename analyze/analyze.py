import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import os
from os.path import join
import pandas as pd

from analyze.metrics.IDNN import estimate
from analyze.metrics import rsa, sum_of_squares


# compute ID, SS and RSA on extracted activations
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

def calc_ID_SS(extract):
    ID = []
    SS = []
    for name, model in tqdm(extract.items()):
        print(name)
        layers = model['layers']
        labels = model['labels']

        ids = []
        ss = []
        for layer in layers:
            id, error = computeID(layer)
            ids.append(id)
            tss, ssw = sum_of_squares.sum_squared(layer, labels)
            ss.append(ssw / tss if tss != 0 else 0)

        ID.append(ids)
        SS.append(ss)

    return ID, SS

def calc_ID_SS_seeds(dataset):
    total_ids = []
    total_ss = []
    for seed in tqdm(range(1, 11)):
        path = join(models_dir, 'models_' + str(seed))
        extract = torch.load(join(path, dataset + '_extracted.pt'))

        seed_ids, seed_ss = calc_ID_SS(extract)
        total_ids += seed_ids
        total_ss += seed_ss

    return total_ids, total_ss


if __name__ == '__main__':
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Devise used = cuda on ", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Devise used = ", device)

    #######
    pre_dataset = 'mnist_noise'
    target_dataset = 'mnist'
    #######

    root_dir = os.getcwd()
    if target_dataset in ['custom3D', 'malaria', 'pets']:
        models_dir = join(root_dir, '../models', 'vgg16', pre_dataset)
    else:
        models_dir = join(root_dir, '../models', pre_dataset)

    # load df
    df_path = join(models_dir, 'df_pre_' + pre_dataset + '+metrics')
    df = pd.read_pickle(df_path)

    # calc ID, SS, RSA and add to df
    print(f'>>> Calculate ID, SS for models pre-trained on {pre_dataset}, on target {target_dataset} <<<')

    if f'ID_{target_dataset}' in df.columns:
        print(df[f'ID_{target_dataset}'])
    if f'SS_{target_dataset}' in df.columns:
        print(df[f'SS_{target_dataset}'])

    if target_dataset in ['custom3D', 'malaria', 'pets']:
        extract = torch.load(join(models_dir, target_dataset + '_extracted.pt'))
        id, ss = calc_ID_SS(extract)
    else:
        id, ss = calc_ID_SS_seeds(target_dataset)
    print(id, ss)
    df[f'ID_{target_dataset}'] = pd.Series(id)
    df[f'SS_{target_dataset}'] = pd.Series(ss)

    if f'RSA_{target_dataset}' in df.columns:
        print(df[f'RSA_{target_dataset}'])

    if target_dataset in ['custom3D', 'malaria', 'pets']:
        rdm_metric = rsa.get_rdm_metric_vgg(pre_dataset, target_dataset)
    else:
        rdm_metric = rsa.get_rdm_metric(pre_dataset, target_dataset)
    df[f'RSA_{target_dataset}'] = pd.Series(rdm_metric)
    # df[f'RSA_{target_dataset}'] = [rdm_metric]   # for VGG

    df.to_pickle(df_path)

    print(' ---- \n>>> df saved with metrics at ', df_path)
