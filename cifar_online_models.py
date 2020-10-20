import torch
import os
from os.path import join
import train_utils
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import datasets

# safety fix seed
train_utils.set_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parse args from sh script
pretrain_dataset = 'imagenet'
dataset_name = 'custom3D'
bs = 1

# set dir
root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)  # target data for ft
source_dir = join(root_dir, 'models', pretrain_dataset)
output_dir = join(source_dir, 'ft_' + dataset_name)  # new folder for fine-tuned models

dataset = datasets.Custom3D(dataset_dir=dataset_dir, device=device)
# dataset = datasets.CIFAR10(dataset_dir=dataset_dir, device=device)

class_names = dataset.class_names
n_out_classes = len(class_names)  # 40 for custom3D
train_loader = dataset.get_train_loader(batch_size=bs, shuffle=False)
test_loader = dataset.get_test_loader(batch_size=bs)


def vgg16_cifar10():
    import torch.nn as nn
    import torch.utils.model_zoo as model_zoo
    from collections import OrderedDict

    def cifar10(n_channel, pretrained=None):
        cfg = [n_channel, n_channel, 'M', 2 * n_channel, 2 * n_channel, 'M', 4 * n_channel, 4 * n_channel, 'M',
               (8 * n_channel, 0), 'M']
        layers = make_layers(cfg, batch_norm=True)
        model = CIFAR(layers, n_channel=8 * n_channel, num_classes=10)
        if pretrained is not None:
            m = model_zoo.load_url(model_urls['cifar10'])
            state_dict = m.state_dict() if isinstance(m, nn.Module) else m
            assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
            model.load_state_dict(state_dict)
        return model




if __name__ == '__main__':
    model = cifar10(128, pretrained='log/cifar10/best-135.pth')
    embed()


## see this link:  https://github.com/aaron-xichen/pytorch-playground/blob/master/cifar/model.py


