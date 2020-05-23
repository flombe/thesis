import os
from os.path import join
import torch
import torch.nn.functional as F
import torch.optim as optim
import mnist_archs
import train_utils
import datasets
#import vgg_mod


# parse args from sh script
parser = train_utils.train_args_parser()
parser.add_argument('--dataset', default='mnist', choices=['mnist', 'mnist2class', 'cifar10'])
args = parser.parse_args()
batch_size, epochs, lr, run_name, seed = train_utils.parse_train_args(args)
dataset_name = args.dataset
print('Run {run_name} on dataset {dataset}'.format(run_name=run_name, dataset=dataset_name))

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)

# set directory
root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)
os.makedirs(dataset_dir, exist_ok=True)
#print('Results directory ' + dataset_dir)


# run every training 10 times to aggregate results for stat testing
for seed_run in range(10):
    ## set seed
    train_utils.set_seed(seed+seed_run)  ## parse starting seed, then add 1 for following runs

    if dataset_name == 'mnist':
        model = mnist_archs.Net()
        model.to(device)
        dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)
        criterion = F.cross_entropy ##

    elif dataset_name == 'mnist2class':
        model = mnist_archs.Net2class()  ## same architecture but only 2 output layers
        model.to(device)
        dataset = datasets.MNIST2class(dataset_dir=dataset_dir, device=device)
        criterion = F.cross_entropy

    else:
        pass
        #model = vgg_mod.vgg16(pretrained=False, num_classes=10)
        #model.to(device)
        #dataset = datasets.CIFAR10(dataset_dir=root_dir, device=device)
        #criterion = F.cross_entropy

    print(model) ## check

    # loaders
    train_loader = dataset.get_train_loader(batch_size=batch_size)
    test_loader = dataset.get_test_loader(batch_size=batch_size)

    # Training
    optimizer = optim.Adam(model.parameters(), lr=lr)  ## Adam instead of SGD
    train_acc, train_loss, test_acc, test_loss = train_utils.train(model=model, train_loader=train_loader,
                                                                   test_loader=test_loader, optimizer=optimizer,
                                                                   device=device, criterion=criterion, epochs=epochs,
                                                                   output_dir=dataset_dir, run_name = run_name,
                                                                   seed = seed+seed_run)
    print('Done trainings run ', seed_run)

print('Done with all training runs.')
