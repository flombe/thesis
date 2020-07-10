import torch
import os
from os.path import join
import argparse
import datasets
import train_utils
from natsort import natsorted


# safety fix seed
train_utils.set_seed(1)

# extract activations on all layers for 500 fixed samples
samples = 500
batch_size = 20  # faster then bs=500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parse args form sh script
parser = argparse.ArgumentParser()
parser.add_argument('--trained_on', default='mnist', choices=['mnist2class'])
parser.add_argument('--dataset', default='mnist', choices=['mnist', 'mnist2class', 'cifar10'])
parser.add_argument('--model_folder', default='all', help='select specific model folder number or all')
args = parser.parse_args()


def get_loader(dataset_name):
    dataset_dir = join(os.getcwd(), 'data', dataset_name)

    if dataset_name == 'mnist':
        dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)
    elif dataset_name == 'mnist2class':
        dataset = datasets.MNIST2class(dataset_dir=dataset_dir, device=device)
    else:
        pass
        # dataset = datasets.CIFAR10(dataset_dir=dataset_dir, device=device)

    test_loader = dataset.get_test_loader(batch_size, shuffle=False)  # using same samples
    return test_loader


def extract(models_dir, test_loader, samples=samples, batch_size=batch_size):

    all_models = {}
    for file in natsorted(os.listdir(models_dir)):  # right order with natsort
        if file.endswith(".pt") and file.startswith("model_"):
            print(file)
            # Load model
            model = torch.load(join(models_dir, file))
            model.to(device)

            with torch.no_grad():
                model.eval()

                all_layers_flattened = []
                i = 0
                labels = []
                for batch in test_loader:
                    # print(batch[0].shape)
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
        for i in range(1, 11):
            models_dir = join(model_dir, 'models_' + str(i))
            extract(models_dir, test_loader=test_loader)
        return print('Extracted all 10 model folders.')
    else:
        models_dir = join(model_dir, 'models_' + str(models))
        extract(models_dir, test_loader=test_loader)
        return print('Extracted all models of folder ', models)


# dataset and dataloader
dataset_name = args.dataset
if type(dataset_name) != str:
    for dataset_name in dataset_name:
        print('Extract for multiple datasets -- dataset = ', dataset_name)
        extract_select(get_loader(dataset_name))

extract_select(get_loader(dataset_name))
