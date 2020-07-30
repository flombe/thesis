import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import os
from os.path import join
import pandas as pd

from IDNN.intrinsic_dimension import estimate, block_analysis
import sum_of_squares
import rsa

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


def calc_ID_SS(dataset):
    total_ids = []
    total_ss = []
    for seed in tqdm(range(1, 11)):  # for 10 seed folders
        path = join(models_dir, 'models_' + str(seed))
        extract = torch.load(join(path, dataset + '_extracted.pt'))  # map_location='cpu'

        seed_ids = []
        seed_ss = []
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

            # df[df['seed']==seed & df['model_name']==name]['target_dataset'] = ids  # add each single value
            seed_ids.append(ids)
            seed_ss.append(ss)

        # df[df['seed'] == seed]['ID_pre'] = seed_ids  # add values for 11 models at once
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


    # extracted on both train and target dataset
    trained_dataset = 'mnist'
    target_dataset = 'fashionmnist_noisy'  # extracted on


    root_dir = os.getcwd()
    models_dir = join(root_dir, 'models', trained_dataset)

    # load df
    df_path = join(models_dir, 'df_pre_' + trained_dataset + '+metrics')
    df = pd.read_pickle(df_path)

    if target_dataset == 'fashionmnist':
        # calc metrics and add to df
        df['target_dataset'] = target_dataset
        print(f'>> Calculate ID, SS for models pre-trained on {trained_dataset}, with target {target_dataset} <<<')
        id_pre, ss_pre = calc_ID_SS(trained_dataset)
        print(id_pre)
        df['ID_pre'] = pd.Series(id_pre)  # add new column and all values at once
        df['SS_pre'] = pd.Series(ss_pre)

        id_target, ss_target = calc_ID_SS(target_dataset)
        df['ID_target'] = pd.Series(id_target)
        df['SS_target'] = pd.Series(ss_target)

        # RSA
        rdm_metric = rsa.get_rdm_metric(trained_dataset, target_dataset)  # standard = euclid dist.
        df['RSA'] = rdm_metric

        df.to_pickle(join(df_path + '+metrics'))  # just to check - Later save as same name
    else:
        id_target, ss_target = calc_ID_SS(target_dataset)
        df['ID_fashion_noisy'] = pd.Series(id_target)
        df['SS_fashion_noisy'] = pd.Series(ss_target)

        # RSA
        rdm_metric = rsa.get_rdm_metric(trained_dataset, target_dataset)  # standard = euclid dist.
        df['RSA_fashion_noisy'] = rdm_metric

        df.to_pickle(join(df_path))
