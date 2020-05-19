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
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,  ## lr hyperparm tune?
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--run_name' , default="pre-train", type=str, metavar='N',
                        help='name of trainings run for saving')
    return parser


def parse_train_args(args):
    bs = args.bs
    epochs = args.epochs
    lr = args.lr
    run_name = args.run_name
    return bs, epochs, lr, run_name


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


def train(model, train_loader, test_loader, optimizer, device, epochs=20, run_name='pre_train', criterion=F.cross_entropy, output_dir=None,  ## F.cross_entropy instead of F.nll_loss
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
            if epoch == 0:
                if i in range(10) or i%100==0:
                    torch.save(model, join(model_dir, 'model_' + str(run_name) + '0batch' +str(i) + '.pt'))

        # save model
        epoch_count = epoch+1  ## epoch count starts at 0
        if model_dir is not None:
            if epoch_count in [1, 2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 100, 150, 200]:
                torch.save(model, join(model_dir, 'model_' + str(run_name) + str(epoch_count) + '.pt'))

        # evaluate model on training and test data
        model.eval()
        loss, acc = evaluate(model, train_loader, device)
        train_loss.append(loss.item())
        train_acc.append(acc)

        loss, acc = evaluate(model, test_loader, device)
        test_loss.append(loss.item())
        test_acc.append(acc)

        if verbose:
            # print statistics
            print('Test loss : %g --- Test acc : %g %%' % (test_loss[-1], test_acc[-1]))

    train_stats = {
        'model_cls': model.__class__.__name__,
        'run_name': run_name,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'test_acc': test_acc,
        'test_loss': test_loss
    }
    stats_file_path = join(output_dir, run_name+'_train_stats.json')  ## otherwise override json files for same model name but different runs
    with open(stats_file_path, 'w+') as f:
        json.dump(train_stats, f)

    return train_loss, train_acc, test_loss, test_acc
