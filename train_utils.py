import os
from os.path import join
from tqdm import tqdm
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# parse args from sh script
def train_args_parser():
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'mnist2class', 'cifar10'],
                        type=str , metavar='D', help='trainings dataset name')
    parser.add_argument('--epochs', default=200, type=int, metavar='E',
                        help='number of total epochs to run')
    parser.add_argument('--bs', '--batch-size', default=64, type=int,
                        metavar='BS', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,  ## lr hyperparm tune?
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--run_name', default="pre-train", type=str, metavar='R',
                        help='trainings run name for saving')
    parser.add_argument('--seeds', default="0", type=int, metavar='S',
                        help='amount of runs with different seeds')
    return parser


def parse_train_args(args):
    dataset = args.dataset
    bs = args.bs
    epochs = args.epochs
    lr = args.lr
    run_name = args.run_name
    seeds = args.seeds
    return dataset, bs, epochs, lr, run_name, seeds


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


### helper functions for tensorboard tracking
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

###


def evaluate(model, loader, device, criterion=F.cross_entropy):  ## F.cross_entropy instead of F.nll_loss
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels)
            total += labels.size(0)

        acc = 100 * correct / total
        loss = loss / total

    return loss, acc


# set checkpoints (for later log-plots choose log ticks)
first_batches_chkpts = [1, 3, 10, 30, 100, 300]
epoch_chkpts = [1, 3, 10, 30, 100]

def train(model, train_loader, test_loader, optimizer, device, epochs, run_name, seed,
          criterion=F.cross_entropy, save=True, output_dir=None, verbose=True):
    model_dir = None
    if output_dir is not None:
        model_dir = join(output_dir, 'models_' + str(seed))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print('Created: Result directory ', model_dir)
        else:
            print("Warning: model directory already exists")

    # tensorboard init
    writer = SummaryWriter(join('tensorboard', run_name, str(seed)))
    running_loss = 0.0

    # training and information collection
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in tqdm(range(epochs)):

        model.train()
        for i, data in enumerate(train_loader, 0):
            model.train()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # tracking for tensorboard
            classes = np.unique(labels.cpu())
            #print('classes: ', classes)

            running_loss += loss.item()
            if i % 100 == 99:  # every 100 batches
                # log the running loss
                writer.add_scalar('training loss', running_loss / 100, epoch * len(train_loader) + i)

                # log a Matplotlib Figure showing the model's predictions on a random mini-batch
                writer.add_figure('predictions vs. actuals',
                                  plot_classes_preds(model, inputs, labels, classes),
                                  global_step=epoch * len(train_loader) + i)
                running_loss = 0.0


            # save models and stats for batches of first epoch
            if save==True and epoch == 0:
                j = i+1
                if j in first_batches_chkpts:
                    torch.save(model, join(model_dir, 'model_' + str(run_name) + '_0_' + str(j) + '.pt'))

                    # save json train stats
                    model.eval()
                    loss, acc = evaluate(model, train_loader, device)
                    train_loss.append(loss.item())
                    train_acc.append(acc)

                    loss, acc = evaluate(model, test_loader, device)
                    test_loss.append(loss.item())
                    test_acc.append(acc)

                    if verbose:
                        print('>>> Epoch 0 - Batch %g :   Test loss : %g --- Test acc : %g %%' % (j, test_loss[-1], test_acc[-1]))


        # evaluate model on training and test data
        model.eval()
        tr_loss, tr_acc = evaluate(model, train_loader, device)
        tst_loss, tst_acc = evaluate(model, test_loader, device)

        if verbose:
            print('Test loss : %g --- Test acc : %g %%' % (test_loss[-1], test_acc[-1]))

        # add tensorboard graphs
        writer.add_scalars('Loss', {'test loss': tst_loss,
                                    'train loss': tr_loss}, epoch)
        writer.add_scalars('Acc', {'test acc': tst_acc,
                                   'train acc': tr_acc}, epoch)
        writer.add_graph(model, inputs)


        # save model
        if save==True:
            epoch_count = epoch+1
            if epoch_count in epoch_chkpts:
                torch.save(model, join(model_dir, 'model_' + str(run_name) + '_' + str(epoch_count) + '.pt'))

                # safe train_stats for checkpoints
                train_loss.append(tr_loss.item())
                train_acc.append(tr_acc)

                test_loss.append(tst_loss.item())
                test_acc.append(tst_acc)

    writer.close()

    if save==True:
        train_stats = {
            'model_cls': model.__class__.__name__,
            'run_name': run_name,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'test_acc': test_acc,
            'test_loss': test_loss
        }
        stats_file_path = join(model_dir, run_name + '_train_stats.json')
        with open(stats_file_path, 'w+') as f:
            json.dump(train_stats, f)

    # create dataframe
    df = pd.DataFrame(train_stats)

    return train_loss, train_acc, test_loss, test_acc
