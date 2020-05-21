import torch
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join
import argparse
import os
import json


def train_args_parser():
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--bs', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,  ## lr hyperparm tune?
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--run_name' , default="pre-train", type=str, metavar='N',
                        help='trainings run name for saving')
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


def train(model, train_loader, test_loader, optimizer, device, epochs=20, run_name='pre_train', criterion=F.cross_entropy, output_dir=None,
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

    batch_train_loss = []
    batch_train_acc = []
    batch_test_loss = []
    batch_test_acc = []

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

            # save details for batches of first epoch
            if epoch == 0:
                i = i+1  ## batch count starts at 0
                if i in [1,3, 10,30, 100,300]:
                    torch.save(model, join(model_dir, 'model_' + str(run_name) + '0batch' +str(i) + '.pt'))

                ## additional json trainings save for batches in first epoch
                model.eval()
                loss, acc = evaluate(model, train_loader, device)
                batch_train_loss.append(loss.item())
                batch_train_acc.append(acc)

                loss, acc = evaluate(model, test_loader, device)
                batch_test_loss.append(loss.item())
                batch_test_acc.append(acc)

        # save model
        epoch_count = epoch+1  ## epoch count starts at 0
        if model_dir is not None:
            if epoch_count in [1,3, 10,30, 100,200]:
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


    ## added additional json save for batches of first epoch
    batch_train_stats = {
        'model_cls': model.__class__.__name__,
        'run_name': run_name,
        'train_acc': batch_train_acc,
        'train_loss': batch_train_loss,
        'test_acc': batch_test_acc,
        'test_loss': batch_test_loss
    }
    stats_file_path = join(output_dir,run_name + '_batch_train_stats.json')
    with open(stats_file_path, 'w+') as f:
        json.dump(batch_train_stats, f)


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
