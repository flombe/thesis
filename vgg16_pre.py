import torch
import os
from os.path import join
import train_utils
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
import collections
import vgg_arch
import caffemodel2pytorch


# safety fix seed
train_utils.set_seed(1)

# set device to gpu using cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def places365_test(model):
    arch = 'vgg16'
    model.eval()

    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the class label
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # load the test image
    img_name = '12.jpg'
    if not os.access(img_name, os.W_OK):
        img_url = 'http://places.csail.mit.edu/demo/' + img_name
        os.system('wget ' + img_url)

    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0)).cuda()
    print(input_img.shape)  ## torch.Size([1, 3, 224, 224])

    plt.imshow(input_img.cpu().squeeze().permute(1, 2, 0))
    plt.show()

    # forward pass
    out = model.forward(input_img)
    pred = F.softmax(out, 1).data.squeeze()
    probs, idx = pred.sort(0, True)
    print(probs, idx)

    print('{} prediction on {}'.format(arch, img_name))
    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

def run_places365_caffe_model():
    pretrain_dataset = 'vgg16/places365'
    root_dir = os.getcwd()
    source_dir = join(root_dir, 'models', pretrain_dataset)

    model = caffemodel2pytorch.Net(
        prototxt=join(source_dir, 'deploy_vgg16_places365.prototxt'),
        weights=join(source_dir, 'vgg16_places365.caffemodel'),
        caffe_proto='https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
    )

    model.cuda()
    model.eval()
    torch.set_grad_enabled(False)

    # # test: to make sure to have right procedure of image normalization and channel reordering
    # image = torch.Tensor(8, 3, 224, 224).cuda()
    # # outputs dict of PyTorch Variables
    # output_dict = model(data=image)
    # print(output_dict)

    # test model
    places365_test(model.to(device))

    return model

def get_vgg16_places365():
    # run to save state_dict use: python -m caffemodel2pytorch VGG_ILSVRC_16_layers.caffemodel
    # then load weights into model arch to be able to save as torch model

    pretrain_dataset = 'vgg16/places365'
    root_dir = os.getcwd()
    source_dir = join(root_dir, 'models', pretrain_dataset)

    model = vgg_arch.vgg16(pretrained=False, num_classes=365)
    print(model)
    model.features = torch.nn.Sequential(collections.OrderedDict(zip(
        ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
         'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
         'pool5'], model.features)))
    # last layer name with additional 'a' (?)
    model.classifier = torch.nn.Sequential(
        collections.OrderedDict(zip(['fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8a'], model.classifier)))

    state_dict = torch.load(join(source_dir, 'state_dict_vgg16_pre_places365.pt'))
    # check output dimension
    print(state_dict['fc8a.weight'].shape)

    model.load_state_dict({l: torch.from_numpy(np.array(v)).view_as(p) for k, v in state_dict.items() for l, p in
                           model.named_parameters() if k in l})

    # test model on places365 sample
    model = model.to(device)
    places365_test(model)

    return model

def vggface_test(model):
    model.eval()

    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
    ])

    # load the class label
    file_name = 'names.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/prlz77/vgg-face.pytorch/master/images/names.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip())
    classes = tuple(classes)

    # load the test image
    img_name = 'Zach+Galifianakis.jpg'
    if not os.access(img_name, os.W_OK):
        img_url = 'http://4.bp.blogspot.com/-GmWic9p-oZo/TVY8_4YuYjI/AAAAAAAABI0/t0jRZAn16B0/s1600/Zach%252BGalifianakis.jpg'
        os.system('wget ' + img_url)

    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0)).cuda()
    print(input_img.shape)  ## torch.Size([1, 3, 224, 224])

    plt.imshow(input_img.cpu().squeeze().permute(1, 2, 0))
    plt.show()

    def decode_predictions(preds, top=5):
        LABELS = classes
        results = []
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            result = [[str(LABELS[i].encode('utf8')), pred[i]] for i in top_indices]
            result.sort(key=lambda x: x[1], reverse=True)
            results.append(result)
        return results

    model = model.to(device)
    out = model.forward(input_img)
    preds = out.cpu().data.numpy()

    results = decode_predictions(preds)
    for prediction in results[0]:
        # prediction = results[0][0][0].replace("b'", "").replace("'", "")
        print('{:.3f} -> {}'.format(prediction[1], prediction[0].replace("b'", "").replace("'", "")))

def get_vgg16_vggface():
    # load online weights into vgg16 model arch
    # weights:  http://www.robots.ox.ac.uk/~albanie/pytorch-models.html
    # arch: http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.py

    model = vgg_arch.vgg16(pretrained=False, num_classes=2622)
    print(model)

    model.features = torch.nn.Sequential(collections.OrderedDict(zip(
        ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
         'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
         'pool5'], model.features)))
    model.classifier = torch.nn.Sequential(
        collections.OrderedDict(zip(['fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8'], model.classifier)))

    state_dict = torch.load(join(source_dir, 'vgg_face_dag.pth'))
    # check output dimension
    print(state_dict['fc8.weight'].shape)

    model.load_state_dict({l: torch.from_numpy(np.array(v)).view_as(p) for k, v in state_dict.items() for l, p in
                           model.named_parameters() if k in l})

    # test model on face sample
    model = model.to(device)
    vggface_test(model)

    return model

def cars_test(model):

    model.eval()
    data_transforms = trn.Compose([
            trn.Resize(224),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # load the class label
    import scipy.io
    mat = scipy.io.loadmat(join(source_dir, 'cars_meta.mat'))  # from devkit on dataset page download
    classes = mat['class_names'][0]

    # load the test image
    img_name = '2013-acura-zdx_100404962_l.jpg'
    if not os.access(img_name, os.W_OK):
        img_url = 'https://images.hgmsites.net/lrg/2013-acura-zdx_100404962_l.jpg'
        os.system('wget ' + img_url)

    img = Image.open(img_name)
    input_img = V(data_transforms(img).unsqueeze(0)).cuda()
    print(input_img.shape)  ## torch.Size([1, 3, 224, 224])

    plt.imshow(input_img.cpu().squeeze().permute(1, 2, 0))
    plt.show()

    # forward pass
    out = model.forward(input_img)

    # pred = F.softmax(out, 1).data.squeeze()
    probs, idx = out[0].cpu().sort(0, True)

    print('VGG-16 on Stanford Cars - prediction on {}'.format(img_name))
    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

def get_vgg16_cars():
    # load from: https://drive.google.com/drive/folders/10_OLCLWZMHnVgHCRYJ3Sqcomc_s9DMp8
    # convert from mxnet to pytorch model with MMdnn

    model = vgg_arch.vgg16(pretrained=False, num_classes=164)  ## !! should be 196 but pre_model only has 164
    print(model)
    model.features = torch.nn.Sequential(collections.OrderedDict(zip(
        ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
         'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
         'pool5'], model.features)))
    model.classifier = torch.nn.Sequential(
        collections.OrderedDict(zip(['fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8'], model.classifier)))

    ## Difficult how to properly adopt .npy file
    # state_dict = np.load(join(source_dir, 'kit_pytorch.npy'), allow_pickle=True)
    # print(type(state_dict), state_dict)

    from importlib.machinery import SourceFileLoader
    mymodule = SourceFileLoader('MainModel', join(source_dir, 'kit_cars.py')).load_module()

    pre_model = torch.load(join(source_dir, 'vgg16cars.pth'))
    # print(pre_model)
    state_dict = pre_model.state_dict()

    # check output dimension
    print(state_dict['fc8.weight'].shape)  ## only 164 output! (should be 196 for cars)

    model.load_state_dict({l: torch.from_numpy(np.array(v)).view_as(p) for k, v in state_dict.items() for l, p in
                           model.named_parameters() if k in l})

    # test model on cars sample
    model = model.to(device)
    cars_test(model)

    return model

def cifar10_test(model):
    model.eval()

    import datasets
    dataset_dir = join(os.getcwd(), 'data', 'cifar10')
    dataset = datasets.CIFAR10(dataset_dir=dataset_dir, device=device)
    test_loader = dataset.get_test_loader(batch_size=1)
    image, label = next(iter(test_loader))

    print(label)
    print(image.shape)  ## torch.Size([1, 3, 224, 224])

    plt.imshow(image.cpu().squeeze().permute(1, 2, 0))
    plt.show()

    # forward pass
    out = model.forward(image.to(device))

    print('VGG-19 on Cifar10 - sample label: {}'.format(label))
    # output the prediction
    _, predicted = torch.max(out.data, 1)
    print('Model prediction: ', predicted)

def get_vgg19_cifar10():
    # from  https://github.com/chengyangfu/pytorch-vgg-cifar10

    import torch.nn as nn
    model = vgg_arch.vgg19(pretrained=False, num_classes=10)
    # print(model)
    model.classifier = nn.Sequential(
        nn.Linear(512, 512),  # pooling reduce cifar dim so much that only 512 let in the end
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 10),
    )
    print(model)

    checkpoint = torch.load(join(source_dir, 'model_best_vgg19_cifar.pth.tar'))
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

    # test model on cifar10 sample
    model = model.to(device)
    cifar10_test(model)

    return model

def get_vgg16bn_segnet():
    # from

    model = vgg_arch.vgg16_bn(pretrained=False, num_classes=10)
    print(model)

    checkpoint = torch.load('/mnt/antares_raid/home/bemmerl/segnet_models/mtan_segnet_without_attention_equal_adam_single_task_0_run_1/model_checkpoints/checkpoint.chk')

    print('--- Layers with key name difference ---')
    for key1, key2 in zip(checkpoint['model_state_dict'].keys(), model.state_dict().keys()):
        if key1 != key2:
            print(key1, checkpoint['model_state_dict'][key1].shape)
            print(key2, model.state_dict()[key2].shape)
            print('-------')

    new_state_dict = dict()
    for key, key2 in zip(checkpoint['model_state_dict'].keys(), model.state_dict().keys()):
        if str(key).startswith('encoder.'):
            new_state_dict[key2] = checkpoint['model_state_dict'][key]  # compared before, change layer names
        else:
            new_state_dict[key2] = model.state_dict()[key2]  # since no classifer in SegNet, append untrained layers

    model.load_state_dict(new_state_dict)
    print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    return model


def create_df_pre(net, pre_dataset, ep, top1, top5):
    pre_stats = {
        'model_name': [f'model_{net}_pre_{pre_dataset}'],
        'pre_net': [net],
        'pre_dataset': [pre_dataset],
        'pre_epochs': [ep],
        'pre_top1': [top1],  # [(100-top1)/100],  if error not acc given
        'pre_top5': [top5]  # [(100-top5)/100]
    }
    # create dataframe
    df = pd.DataFrame(pre_stats)
    df.to_pickle(join(source_dir, f'df_pre_{pre_dataset}'))
    return df


if __name__=='__main__':

    init_places365 = False
    init_imagenet = False
    init_vggface = False
    init_cars = False
    init_cifar10 = False
    init_segnet = False

    ## places365
    if init_places365:
        pretrain_dataset = 'vgg16/places365'
        source_dir = join(os.getcwd(), 'models', pretrain_dataset)

        # get weights from caffe model, init pytorch model arch, load weights and test on places365 sample
        model = get_vgg16_places365()
        torch.save(model, join(source_dir, 'model_vgg16_pre_places365.pt'))

        # create and save df with online available pre-train data (https://github.com/CSAILVision/places365)
        df = create_df_pre(net='vgg16', pre_dataset='places365', ep=90, top1=55.19, top5=85.01)

        # run finetune_vgg16.sh to ft on custom3D

    # imagenet
    if init_imagenet:
        pretrain_dataset = 'vgg16/imagenet'
        source_dir = join(os.getcwd(), 'models', pretrain_dataset)

        model_pre = vgg_arch.vgg16(pretrained=True)  # pre-trained on imageNet
        torch.save(model_pre, join(source_dir, 'model_vgg16_pre_imagenet.pt'))

        df = create_df_pre(net='vgg16', pre_dataset='imagenet', ep=74,
                           top1=(100 - 55.19) / 100,
                           top5=(100 - 85.01) / 100)  # top-err given not acc

    ## VGGFace
    if init_vggface:
        # http://www.robots.ox.ac.uk/~albanie/pytorch-models.html
        # http://www.robots.ox.ac.uk/~vgg/software/vgg_face

        pretrain_dataset = 'vgg16/vggface'
        source_dir = join(os.getcwd(), 'models', pretrain_dataset)

        model = get_vgg16_vggface()
        torch.save(model, join(source_dir, 'model_vgg16_pre_vggface.pt'))

        # create and save df with online available pre-train data
        df = create_df_pre(net='vgg16', pre_dataset='vggface', ep='', top1=0.9722, top5='')

        # run finetune_vgg16.sh to ft on custom3D

    ## Cars
    if init_cars:
        # Cars dataset contains 16,185 images of 196 classes of cars. https://ai.stanford.edu/~jkrause/cars/car_dataset.html
        # https://github.com/afifai/car_recognizer_aiforsea

        pretrain_dataset = 'vgg16/cars'
        source_dir = join(os.getcwd(), 'models', pretrain_dataset)

        model = get_vgg16_cars()
        torch.save(model, join(source_dir, 'model_vgg16_pre_cars.pt'))

        # create and save df with online available pre-train data
        df = create_df_pre(net='vgg16', pre_dataset='cars', ep='65', top1=0.853, top5=0.959)

        # run finetune_vgg16.sh to ft on custom3D

    ## Cifar10
    if init_cifar10:
        # https://github.com/geifmany/cifar-vgg

        pretrain_dataset = 'vgg16/cifar10'
        source_dir = join(os.getcwd(), 'models', pretrain_dataset)

        # model = get_vgg19_cifar10()
        # torch.save(model, join(source_dir, 'model_vgg19_pre_cifar10.pt'))

        model = torch.load(join(source_dir, 'model_vgg19_pre_cifar10.pt'))
        model = model.to(device)
        cifar10_test(model)

        # create and save df with online available pre-train data
        df = create_df_pre(net='vgg19', pre_dataset='cifar10', ep='233', top1=0.9243, top5='')

        # run finetune_vgg16.sh to ft on custom3D

    ## SegNet
    if init_segnet:
        # http://mi.eng.cam.ac.uk/projects/segnet/
        # Bachelor Thesis at NI

        pretrain_dataset = 'vgg16/segnet'
        source_dir = join(os.getcwd(), 'models', pretrain_dataset)

        model = get_vgg16bn_segnet()
        torch.save(model, join(source_dir, 'model_vgg16bn_pre_segnet.pt'))

        # model test not useful, since classifier not trained (since not loaded from pre)

        # create and save df with online available pre-train data
        df = create_df_pre(net='vgg16bn', pre_dataset='camvid', ep='99', top1=0.9857, top5='')






# VGG(
#   (features): Sequential(
#     (conv1_1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu1_1): ReLU(inplace=True)
#     (conv1_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu1_2): ReLU(inplace=True)
#     (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (conv2_1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu2_1): ReLU(inplace=True)
#     (conv2_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu2_2): ReLU(inplace=True)
#     (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (conv3_1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu3_1): ReLU(inplace=True)
#     (conv3_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu3_2): ReLU(inplace=True)
#     (conv3_3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu3_3): ReLU(inplace=True)
#     (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (conv4_1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu4_1): ReLU(inplace=True)
#     (conv4_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu4_2): ReLU(inplace=True)
#     (conv4_3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu4_3): ReLU(inplace=True)
#     (pool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (conv5_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu5_1): ReLU(inplace=True)
#     (conv5_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu5_2): ReLU(inplace=True)
#     (conv5_3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (relu5_3): ReLU(inplace=True)
#     (pool5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#  )
#   (classifier): Sequential(
#     (fc6): Linear(in_features=25088, out_features=4096, bias=True)
#     (relu6): ReLU(inplace=True)
#     (drop6): Dropout(p=0.5, inplace=False)
#     (fc7): Linear(in_features=4096, out_features=4096, bias=True)
#     (relu7): ReLU(inplace=True)
#     (drop7): Dropout(p=0.5, inplace=False)
#     (fc8a): Linear(in_features=4096, out_features=365, bias=True)
#   )
# )
