import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from os.path import join
import datasets
import train_utils
import mnist_archs


# parse args from sh script
parser = train_utils.train_args_parser()
parser.add_argument('--ft')  ## add parser arg ft for fine-tune model selection
args = parser.parse_args()
dataset_name, batch_size, epochs, lr, run_name, seeds = train_utils.parse_train_args(args)
ft_source = args.ft
print('Run {run_name} on dataset {dataset} from source {ft_source}.'.format(
    run_name=run_name, dataset=dataset_name, ft_source=ft_source))

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)

# set dir
root_dir = os.getcwd()
dataset_dir = join(root_dir, 'data', dataset_name)  ## MNIST
source_dir = join(root_dir, 'data', ft_source, 'models')  ## MNIST2class


for seed in range(1,11):
    model_dir = join(source_dir +'_'+ str(seed))
    print(model_dir)

    for file in os.listdir(model_dir):
        if file.endswith(".pt"):
            print(file)
            # Load model
            model = torch.load(join(model_dir, file))
            #print(model)
            #print(file[16:-3]) # add source model name for saving
            run_name_sub = join(run_name + file[17:-3] + '_')

            ## print weights to check if change
            #print(list(model.fc1.parameters()))
            #print(list(model.fc2.parameters()))

            # new fine-tune layers
            model_ft = model
            model_ft.fc1 = mnist_archs.Net().fc1  ## = nn.Linear(1600, 128)
            model_ft.fc2 = mnist_archs.Net().fc2  ## = nn.Linear(128, 10)

            ## print newly initalized weights
            ##print(list(model_ft.fc1.parameters()))
            #print(list(model_ft.fc2.parameters()))

            model_ft = model_ft.to(device)

            ## from train
            #if dataset_name == 'mnist':
            dataset = datasets.MNIST(dataset_dir=dataset_dir, device=device)
            criterion = F.cross_entropy

            # loaders
            train_loader = dataset.get_train_loader(batch_size=batch_size)
            test_loader = dataset.get_test_loader(batch_size=batch_size)

            # Training
            optimizer = optim.Adam(model_ft.parameters(), lr=lr)  ## Adam instead of SGD
            train_acc, train_loss, test_acc, test_loss = train_utils.train(model=model_ft, train_loader=train_loader,
                                                                           test_loader=test_loader, optimizer=optimizer,
                                                                           device=device, criterion=criterion, epochs=epochs,
                                                                           output_dir=dataset_dir, run_name = run_name_sub,
                                                                           seed=seed)
            print('Done ', file)

    print('Done all models of seed ', seed)

print('Done fine-tuning all models for all seeds.')
