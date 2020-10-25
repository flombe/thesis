import torch
import os
from os.path import join
import train_utils
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import datasets
import vgg_arch

# safety fix seed
train_utils.set_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parse args from sh script
pretrain_dataset = 'vgg16/cifar10'
dataset_name = 'cifar10'
bs = 1

# set dir
root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)  # target data for ft
source_dir = join(root_dir, 'models', pretrain_dataset)

# dataset = datasets.Custom3D(dataset_dir=dataset_dir, device=device)
dataset = datasets.CIFAR10(dataset_dir=dataset_dir, device=device)

test_loader = dataset.get_test_loader(batch_size=bs)

import torch.nn as nn
model = vgg_arch.vgg19(pretrained=False, num_classes=10)
#print(model)
model.classifier = nn.Sequential(
            nn.Linear(512, 512),     # pooling reduce cifar dim so much that only 512 let in the end
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 10),
        )
print(model)

def vgg16_cifar10():

    if os.path.isfile(join(source_dir, 'model_best.pth.tar')):
        print("=> loading checkpoint")
        checkpoint = torch.load(join(source_dir, 'model_best.pth.tar'))
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        print(start_epoch, best_prec1)
        # print(checkpoint['state_dict'])

        new_state_dict = dict()
        for key in checkpoint['state_dict'].keys():
            # print(key, checkpoint['state_dict'][key].shape)
            if str(key).startswith('features.'):
                print(str(key[:8] + key[15:]))
                new_state_dict[str(key[:8] + key[15:])] = checkpoint['state_dict'][key]
            else:
                new_state_dict[key] = checkpoint['state_dict'][key]

        newer_state_dict = dict()
        print('--- Layers with key name difference ---')
        for key1, key2 in zip(new_state_dict.keys(), model.state_dict().keys()):
            if key1 == key2:
                newer_state_dict[key1] = new_state_dict[key1]
            else:
                print(key1, new_state_dict[key1].shape)
                print(key2, model.state_dict()[key2].shape)
                print('-------')
                newer_state_dict[key2] = new_state_dict[key1]


        model.load_state_dict(newer_state_dict)
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        torch.save(model, join(source_dir, 'model_vgg19_pre_cifar10.pt'))



if __name__ == '__main__':
    vgg16_cifar10()


