import torch
import os
from os.path import join
import argparse
import json
import datasets
import train_utils

# set seed
train_utils.set_seed(1)

samples = 500
batch_size = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', choices=['mnist', 'mnist2class', 'cifar10'])
parser.add_argument('--model_folder', default='all', help='select specific model folder number or all')
args = parser.parse_args()


# dataset and dataloader
dataset_name = args.dataset
dataset_dir = join(os.getcwd(), 'data', dataset_name)

if dataset_name == 'mnist':
    dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)
elif dataset_name == 'mnist2class':
    dataset = datasets.MNIST2class(dataset_dir=dataset_dir, device=device)
else:
    pass
    #dataset = datasets.CIFAR10(dataset_dir=dataset_dir, device=device)

test_loader = dataset.get_test_loader(batch_size, shuffle=False)  #using same samples
imgs,labels= next(iter(test_loader))
print(imgs.shape, labels[0:10])

def extract(models_dir, samples=samples, batch_size=batch_size, test_loader=test_loader):

    all_models = {}
    for file in os.listdir(models_dir):  # !order not in training-logic but dir-alphabetical
        if file.endswith(".pt"):
            print(file)
            # Load model
            model = torch.load(join(models_dir, file))
            model.to(device)

            all_layers_flattened = []
            i = 0
            labels = []
            for batch in test_loader:
                #print(batch[0].shape)
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

    torch.save(all_models, join(models_dir, '_extracted.pt'))
    return print('All models extracted - result dim=', len(all_models))


# model folder
models = args.model_folder
if models == 'all':
    for i in range(1, 11):
        models_dir = join(dataset_dir, 'models_' + str(i))
        extract(models_dir)
    print('Extracted all 10 model folders.')
else:
    models_dir = join(dataset_dir, 'models_' + str(models))
    extract(models_dir)
    print('Extracted all models of folder ', models)


