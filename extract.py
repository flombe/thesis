import torch
import os
from os.path import join
import argparse
import datasets


samples = 512
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10'])
args = parser.parse_args()

dataset_name = args.dataset
dataset_dir = join(os.getcwd(), 'data', dataset_name)
models_dir = join(dataset_dir, 'models')

if dataset_name == 'mnist':
    dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)
else:
    pass
    #dataset = datasets.CIFAR10(dataset_dir=dataset_dir, device=device)

test_loader = dataset.get_test_loader(batch_size)

all_models = {}
for file in os.listdir(models_dir):
    print(file)
    # Load model
    model = torch.load(join(models_dir, file))
    model.to(device)

    all_layers_flattened = []
    i = 0
    labels = []
    for batch in test_loader:
        print(batch[0].shape)
        all_layers = [batch[0]] + list(model.extract_all(batch[0].to(device), verbose=False))
        all_layers_flattened.append([out.view(batch_size, -1).cpu().data for out in all_layers])
        labels.append(batch[1].data)
        if i == int(samples / batch_size):
            break
        i += 1

    result = []
    for layers in zip(*all_layers_flattened):
        result.append(torch.cat(layers))

    all_models[file] = {
        'layers': result,
        'labels': torch.cat(labels)
    }

torch.save(all_models, join(dataset_dir, 'extracted.pt'))
