import torch
import os
from os.path import join
import argparse
import datasets
import train_utils
from natsort import natsorted
from tqdm import tqdm
import numpy as np

# safety fix seed
train_utils.set_seed(1)

# extract activations on all layers for 500 fixed samples
samples = 500
batch_size = 1
unique_labels = 10  # for mnist like datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parse args form sh script
parser = argparse.ArgumentParser()
parser.add_argument('--trained_on', default='mnist',
                    choices=['mnist', 'mnist2class', 'fashionmnist', 'mnist_noise_struct', 'mnist_noise'])
parser.add_argument('--dataset', default='mnist',
                    choices=['mnist', 'mnist2class', 'fashionmnist', 'mnist_noise_struct', 'mnist_noise', 'cifar10'])
parser.add_argument('--model_folder', default='all', help='select specific model folder number or all')
args = parser.parse_args()


def get_loader(dataset_name):
    dataset_dir = join(os.getcwd(), 'data', dataset_name)

    if dataset_name == 'mnist':
        dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)
    elif dataset_name == 'mnist2class':
        dataset = datasets.MNIST2class(dataset_dir=dataset_dir, device=device)
    elif dataset_name == 'mnist_noise':
        dataset = datasets.MNIST_noise(dataset_dir=dataset_dir, device=device)
    elif dataset_name == 'fashionmnist':
        dataset = datasets.FashionMNIST(dataset_dir=dataset_dir, device=device)
    elif dataset_name == 'mnist_noise_struct':
        dataset = datasets.MNIST_noise_struct(dataset_dir=dataset_dir, device=device)
    else:
        # dataset_dir = join(os.getcwd(), 'data', 'cifar-10-batches-py')  ###
        dataset = datasets.CIFAR10(dataset_dir=dataset_dir, device=device)

    test_loader = dataset.get_test_loader(batch_size, shuffle=False)  # using same samples
    return test_loader


def extract(models_dir, test_loader, samples=samples, batch_size=batch_size, balanced=True):

    all_models = {}
    for file in natsorted(os.listdir(models_dir)):  # right order with natsort
        if file.endswith(".pt") and file.startswith("model_"):
            print(file)
            # Load model
            model = torch.load(join(models_dir, file))
            model.to(device)

            with torch.no_grad():
                model.eval()

                i = 0
                all_layers_flattened = []
                labels = []
                if balanced:
                    class_counts = {}
                    k = samples / unique_labels  # nr of samples per class
                    for batch in test_loader:  ## set batch_size = 1!
                        c = batch[1].item()  # get label
                        class_counts[c] = class_counts.get(c, 0) + 1  # class count dict
                        if class_counts[c] <= k:
                            all_layers = [batch[0]] + list(model.extract_all(batch[0].to(device), verbose=False))
                            all_layers_flattened.append([out.view(batch_size, -1).cpu().data for out in all_layers])
                            labels.append(batch[1].data)
                        if len(labels) == samples:
                            break
                    # print('labels & count: ', np.unique(np.array(labels), return_counts=True))
                else:
                    for batch in test_loader:
                        all_layers = [batch[0]] + list(model.extract_all(batch[0].to(device), verbose=False))
                        all_layers_flattened.append([out.view(batch_size, -1).cpu().data for out in all_layers])
                        labels.append(batch[1].data)
                        if i == int(samples / batch_size)-1:
                            break
                        i += 1

                result = []
                for layers in zip(*all_layers_flattened):
                    result.append(torch.cat(layers))

            all_models[file] = {
                'layers': result,
                'labels': torch.cat(labels)
            }

    torch.save(all_models, join(models_dir, dataset_name + '_extracted.pt'))
    return print('All models extracted - result dim=', len(all_models))


def extract_select(test_loader):
    # model folder
    model_dir = join(os.getcwd(), 'models', args.trained_on)
    models = args.model_folder
    if models == 'all':
        for i in tqdm(range(1, 11)):
            models_dir = join(model_dir, 'models_' + str(i))
            extract(models_dir, test_loader=test_loader)
        return print('Extracted all 10 model folders.')
    else:
        models_dir = join(model_dir, 'models_' + str(models))
        extract(models_dir, test_loader=test_loader)
        return print('Extracted all models of folder ', models)


# dataset and dataloader
dataset_name = args.dataset
extract_select(get_loader(dataset_name))  ## Balanced Dataset set TRUE as standard
