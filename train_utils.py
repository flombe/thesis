import torch
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join
import argparse
import os
import json




def train_args_parser():
    parser = argparse.ArgumentParser(description='MNIST Experiment')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--bs', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    return parser


def parse_train_args(args):
    bs = args.bs
    epochs = args.epochs
    lr = args.lr
    momentum = args.momentum
    return bs, epochs, lr, momentum


def evaluate(model, loader, device='cuda', criterion=F.nll_loss):
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


def train(model, train_loader, test_loader, optimizer, device='cuda', epochs=20, criterion=F.nll_loss, output_dir=None,
          verbose=True):
    model_dir = None
    if output_dir is not None:
        model_dir = join(output_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            print("Warning: model directory already exists")

    # training and information collection
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in tqdm(range(epochs)):

        model.train()
        for i, data in enumerate(train_loader, 0):
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

        # save model
        if model_dir is not None:
            torch.save(model, join(model_dir, 'model_' + str(epoch) + '.pt'))

        # evaluate model on training and test data
        model.eval()
        loss, acc = evaluate(model, train_loader)
        train_loss.append(loss.item())
        train_acc.append(acc)

        loss, acc = evaluate(model, test_loader)
        test_loss.append(loss.item())
        test_acc.append(acc)

        if verbose:
            # print statistics
            print('Test loss : %g --- Test acc : %g %%' % (test_loss[-1], test_acc[-1]))

    train_stats = {
        'model_cls': model.__class__.__name__,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'test_acc': test_acc,
        'test_loss': test_loss
    }
    stats_file_path = join(output_dir, 'train_stats.json')
    with open(stats_file_path, 'w+') as f:
        json.dump(train_stats, f)

    return train_loss, train_acc, test_loss, test_acc
