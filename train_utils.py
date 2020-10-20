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
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'mnist2class', 'mnist_noise',
                                                               'mnist_noise_struct', 'mnist_split1', 'mnist_split2',
                                                               'fashionmnist', 'cifar10', 'imagenet', 'custom3D',
                                                               'random_init', 'places365'],
                        type=str , metavar='D', help='trainings dataset name')
    parser.add_argument('--epochs', default=100, type=int, metavar='E',
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
        plt.imshow(np.transpose(npimg.astype('uint8'), (1, 2, 0)))


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


def evaluate(model, loader, device, criterion=F.nll_loss):
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
            loss += criterion(outputs, labels) * labels.size(0)  ## error in ID paper? criterion gives loss mean
            total += labels.size(0)

        acc = 100 * correct / total
        loss = loss / total

    return loss, acc


# set checkpoints (for later log-plots choose log ticks)
first_batches_chkpts = np.array([0, 1, 3, 10, 30, 100, 300])
epoch_chkpts = np.array([1, 3, 10, 30, 100])
bs_factor = 0.001

def train(model, train_loader, test_loader, optimizer, device, epochs, run_name, seed,
          criterion=F.nll_loss, save=True, ft=False, output_dir=None, verbose=True, scheduler=False):

    # for custom3D - bs=12 --> different batch epoch split
    if model.__class__.__name__ == 'VGG':
        print('Using lr scheduler for training.')
        first_batches_chkpts = np.array([0, 1, 3, 10, 30])
        bs_factor = 0.01

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

    # training and information collection
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    model_names = []


    ## added untrained model (as benchmark)
    if save == True:
        if 0 in first_batches_chkpts:
            model_names.append(join('model_' + str(run_name) + '_0.pt'))
            torch.save(model, join(model_dir, model_names[-1]))

            # save train stats
            model.eval()

            loss0, acc0 = evaluate(model, train_loader, device)
            train_loss.append(loss0.item())
            train_acc.append(acc0)

            loss0, acc0 = evaluate(model, test_loader, device)
            test_loss.append(loss0.item())
            test_acc.append(acc0)


    for epoch in tqdm(range(epochs)):

        run_train_loss = 0.0
        # run_train_acc = 0.0
        run_sample_nr = 0

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


            # last batch is only 32 samples!! 60K/64 = 937.5 -> train_loader 0-936 with 64, but 937 only 32
            run_train_loss += loss.item() * inputs.size(0)  # loss mean of batch * batch size (for last one diff)
            # run_train_acc += (torch.argmax(outputs, 1) == labels).float().sum()  # too rough estimate
            run_sample_nr += inputs.size(0)

            if i % 100 == 99:  # every 100 batches
                # log the running loss
                writer.add_scalar('training loss', run_train_loss / run_sample_nr, epoch * len(train_loader) + i)

                ## log a Matplotlib Figure showing the model's predictions on a random mini-batch
                # classes = np.unique(labels.cpu())
                # writer.add_figure('predictions vs. actuals',
                #                   plot_classes_preds(model, inputs, labels, classes),
                #                   global_step=epoch * len(train_loader) + i)

            # save models and stats for batches of first epoch
            if save==True and epoch == 0:
                j = i+1
                if j in first_batches_chkpts:
                    model_names.append(join('model_' + str(run_name) + '_0_' + str(j) + '.pt'))
                    torch.save(model, join(model_dir, model_names[-1]))

                    # save train stats
                    model.eval()

                    loss, acc = evaluate(model, train_loader, device)
                    train_loss.append(loss.item())
                    train_acc.append(acc)

                    loss, acc = evaluate(model, test_loader, device)
                    test_loss.append(loss.item())
                    test_acc.append(acc)

                    if verbose:
                        print('>>> Epoch 0 - Batch %g :   Test loss : %g --- Test acc : %g %%' % (j, test_loss[-1], test_acc[-1]))

        # # running loss/acc on training as estimate
        # run_train_loss /= run_sample_nr
        # run_train_acc /= run_sample_nr


        # evaluate model on training and test data
        if seed == 1:  ## to speed up training of other rounds, only track first in detail
            model.eval()
            tr_loss, tr_acc = evaluate(model, train_loader, device)
            tst_loss, tst_acc = evaluate(model, test_loader, device)

        # save model
        if save==True:
            epoch_count = epoch+1
            if epoch_count in epoch_chkpts:
                model_names.append(join('model_' + str(run_name) + '_' + str(epoch_count) + '.pt'))
                torch.save(model, join(model_dir, model_names[-1]))

                if seed != 1:  # only calculate at checkpts
                    model.eval()
                    tr_loss, tr_acc = evaluate(model, train_loader, device)
                    tst_loss, tst_acc = evaluate(model, test_loader, device)

                # safe train_stats for checkpoints
                train_loss.append(tr_loss.item())
                train_acc.append(tr_acc)

                test_loss.append(tst_loss.item())
                test_acc.append(tst_acc)

        if verbose:
            print('Test loss : %g --- Test acc : %g %%' % (tst_loss, tst_acc))

        # add tensorboard graphs
        writer.add_scalars('Loss', {'test loss': tst_loss,
                                    'train loss': tr_loss}, epoch)
        writer.add_scalars('Acc', {'test acc': tst_acc,
                                   'train acc': tr_acc}, epoch)
        writer.add_graph(model, inputs)

        if scheduler != False:
            scheduler.step()  # pytorch scheduler step after optimizer
            print(scheduler.get_last_lr())

    writer.close()

    # save training or finetuning stats dict to df
    if save == True:
        if ft == False:
            train_stats = {
                'model_name': model_names,
                'seed': seed,
                'pre_net': model.__class__.__name__,
                'pre_epochs': np.append(first_batches_chkpts * bs_factor, epoch_chkpts).tolist(),  ##
                'pre_train_acc': train_acc,
                'pre_train_loss': train_loss,
                'pre_test_acc': test_acc,
                'pre_test_loss': test_loss
            }
        else:  # save == True & ft == True
            train_stats = {
                'model_name': model_names,
                'seed': seed,
                'ft_net': model.__class__.__name__,
                'ft_epochs': np.append(first_batches_chkpts * bs_factor, epoch_chkpts).tolist(),  ##
                'ft_train_acc': train_acc,
                'ft_train_loss': train_loss,
                'ft_test_acc': test_acc,
                'ft_test_loss': test_loss
            }

        stats_file_path = join(model_dir, run_name + '_train_stats.json')
        with open(stats_file_path, 'w+') as f:
            json.dump(train_stats, f)

        # create dataframe
        df = pd.DataFrame(train_stats)

    return train_loss, train_acc, test_loss, test_acc, df
