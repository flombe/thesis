import torch
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.variable import Variable
from natsort import natsorted
import random

import datasets


def add_gaussian(ins, mean, stddev, factor=1):
    '''
       Add gaussian sampled random values to input
    '''
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + factor * noise


def test_gaussian():
    root_dir = os.getcwd()
    dataset_dir = join(root_dir, '../data', 'fashionmnist')

    dataset = datasets.FashionMNIST(dataset_dir=dataset_dir, device=device)
    test_loader = dataset.get_test_loader(batch_size=16)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    img = images[0]
    print(type(img), img.shape)
    print('mean: ', torch.mean(img), 'std: ', torch.std(img))
    plt.imshow(img.squeeze(), cmap="Greys")
    plt.show()

    # add noise
    noisy = add_gaussian(img.squeeze(), 0, 1, 0.5)  # only add 0.5 of noise
    print(noisy.shape)
    print('mean: ', torch.mean(noisy), 'std: ', torch.std(noisy))
    plt.imshow(noisy, cmap="Greys")
    plt.show()

    # all noise
    rand = add_gaussian(torch.zeros(28, 28), 0, 1)
    print(rand.shape)
    print('mean: ', torch.mean(rand), 'std: ', torch.std(rand))
    plt.imshow(rand.numpy(), cmap="Greys")
    plt.show()


def augmentation(batch, arg):
    # print(type(batch), batch[0].shape, batch[1].shape)  # list, torch.Size([1, 1, 28, 28]) torch.Size([1])

    if arg == 'shuffle':
        batch[1] = torch.tensor([random.randint(0, 9)])

    if arg == 'noisy':
        batch[0] = add_gaussian(batch[0], 0, 1, 0.5)  # add 0.5 gaussian

    if arg == 'pure_noise':
        batch[0] = add_gaussian(torch.zeros(1, 1, 28, 28), 0, 1)  # pure noise

    # print(type(batch), batch[0].shape, batch[1].shape)
    return batch


unique_labels = 10
def extract_aug(models_dir, test_loader, samples, batch_size, arg, balanced=True):
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
                labels = []
                if balanced:
                    class_counts = {}
                    k = samples / unique_labels  # nr of samples per class
                    for batch in test_loader:  ## set batch_size = 1!
                        c = batch[1].item()  # get label
                        class_counts[c] = class_counts.get(c, 0) + 1  # class count dict
                        if class_counts[c] <= k:

                            batch = augmentation(batch, arg)
                            # if len(labels)==0: # only once, check augmentation
                            #     plt.imshow(batch[0].squeeze(), cmap="Greys")
                            #     plt.title(str(batch[1]))
                            #     plt.show()

                            all_layers = [batch[0]] + list(model.extract_all(batch[0].to(device), verbose=False))
                            all_layers_flattened.append([out.view(batch_size, -1).cpu().data for out in all_layers])
                            labels.append(batch[1].data)
                        if len(labels) == samples:
                            break
                    # print('labels & count: ', np.unique(np.array(labels), return_counts=True))
                else:
                    pass

                result = []
                for layers in zip(*all_layers_flattened):
                    result.append(torch.cat(layers))

            all_models[file] = {
                'layers': result,
                'labels': torch.cat(labels)
            }

    torch.save(all_models, join(models_dir, dataset_name + '_' + arg + '_extracted.pt'))
    return print('All models extracted - result dim=', len(all_models))



if __name__ is '__main__':
    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Devise used = cuda on ", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Devise used = ", device)

    # test_gaussian()

    # run extract on augmented inputs
    root_dir = os.getcwd()
    dataset_dir = join(root_dir, '../data', 'fashionmnist')

    dataset = datasets.FashionMNIST(dataset_dir=dataset_dir, device=device)
    test_loader = dataset.get_test_loader(batch_size=1, shuffle=False)
    dataset_name = 'fashionmnist'

    for seed in range(1, 11):
        models_dir = join(root_dir, '../models', 'mnist', 'models_' + str(seed))

        #extract_aug(models_dir, test_loader, samples=500, batch_size=1, arg='shuffle')
        #extract_aug(models_dir, test_loader, samples=500, batch_size=1, arg='noisy')
        #extract_aug(models_dir, test_loader, samples=500, batch_size=1, arg='pure_noise')
