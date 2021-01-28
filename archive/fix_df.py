import torch
import os
from os.path import join

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Devise used = cuda on ", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Devise used = ", device)


dataset_name = 'fashionmnist'

root_dir = os.getcwd()
dataset_dir = join(root_dir, '../data', dataset_name)
model_dir = join(root_dir, '../models', dataset_name)

# # to add multiple seed runs (e.g. on random_init) into one df
# df = pd.DataFrame()
# for seed in range(1, 4):
#     dff = pd.read_pickle(f'/mnt/antares_raid/home/bemmerl/thesis/models/vgg16/random_init/models_{seed}/df_pre_random_init+metrics')
#     # dff['seed'] = seed
#     dff.insert(1, 'seed', seed)
#     df = df.append(dff, ignore_index=True)
# df.to_pickle(join(model_dir, f'df_pre_random_init+metrics'))
#


# df_all = pd.DataFrame()
# for seed in range(1, 11):
#     df = pd.read_json(f'/mnt/antares_raid/home/bemmerl/thesis/models/fashionmnist/models_{seed}/pre_fashion_add0_train_stats.json')
#     dff = pd.read_json(f'/mnt/antares_raid/home/bemmerl/thesis/models/fashionmnist/models_{seed}/pre_fashion_train_stats.json')
#
#     df = df.append(dff, ignore_index=True)
#     df_all = df_all.append(df, ignore_index=True)
# df_all.to_pickle(join(model_dir, f'df_pre_fashionmnist'))


# df = pd.DataFrame()
# source_dir = join(root_dir, 'models/fashionmnist/ft_mnist')
# for seed in range(1, 11):
#     model_dir = join(source_dir, 'models_' + str(seed))
#     df_seed = pd.DataFrame()
#     # print(natsorted(os.listdir(model_dir)))  # doesn't put _0 first
#     file_list = natsorted(os.listdir(model_dir))[0:12]
#     file_list.insert(0, file_list.pop(6))  # fix sorting manually
#     for file in file_list:
#         if file.endswith("train_stats.json"):
#             print(file)
#             df_seed = df_seed.append(pd.read_json(join(model_dir, file)), ignore_index=True)
#     df = df.append(df_seed, ignore_index=True)
# df.insert(3, 'ft_dataset', 'mnist')
# param = {'train_samples': 60032,
#          'batch_size': 64,
#          'lr': 0.0001}
# df.insert(4, 'ft_param', [param] * len(df))
# df.to_pickle(join(source_dir, f'df_ft_fashionmnist_mnist'))


# df = pd.DataFrame()
# source_dir = join(root_dir, 'models/mnist_noise_struct/ft_fashionmnist')
# for seed in range(1, 11):
#     print(seed)
#     model_dir = join(source_dir, 'models_' + str(seed))
#     df_seed = pd.DataFrame()
#     # print(natsorted(os.listdir(model_dir)))  # doesn't put _0 first
#     file_list = natsorted(os.listdir(model_dir))[0:12]
#     file_list.insert(0, file_list.pop(6))  # fix sorting manually
#     for file in file_list:
#         if file.endswith("train_stats.json"):
#             print(file)
#             df_seed = df_seed.append(pd.read_json(join(model_dir, file)), ignore_index=True)
#     df = df.append(df_seed, ignore_index=True)
#
# df.insert(3, 'ft_dataset', 'mnist')
# param = {'train_samples': 60032,
#          'batch_size': 64,
#          'lr': 0.0001}
# df.insert(4, 'ft_param', [param] * len(df))
# df.to_pickle(join(source_dir, f'df_ft_mnist_noise_struct_fashionmnist'))


# ## rename the model_names (error while ft, not changed name)
# df = pd.read_pickle('/mnt/antares_raid/home/bemmerl/thesis/models/mnist_noise_struct/ft_fashionmnist/df_ft_mnist_noise_struct_fashionmnist')
# for i in range(len(df)):
#     # print(df['model_name'][i])
#     check = df['model_name'][i].split('model_ft_mnist_noise_struct_mnist', 1)[1]
#     # print(check)
#     # print(str('model_ft_mnist_noise_struct_fashionmnist' + check))
#     df['model_name'][i] = str('model_ft_mnist_noise_struct_fashionmnist' + check)
#     print(df['model_name'][i])
# df.to_pickle('/mnt/antares_raid/home/bemmerl/thesis/models/mnist_noise_struct/ft_fashionmnist/df_ft_mnist_noise_struct_fashionmnist')
