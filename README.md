# Transferability in DNN
#### Master Thesis - Florian Bemmerl - TU Berlin

For details and background to all methods, architectures and datasets used, please refer to the thesis document (thesis.pdf). 

_____

This repository includes python implementation to train, fine-tune, analyze and visualize CNN models for their
transferability performance. 

It is structured into the folders:
 - data: dataset image samples
 - models: saving directory for the model checkpoints during training
 - plots: saving dir. for visualizations
   
 - datasets: scripts to define dataset classes and to create custom datasets
 - train: scripts to run training and fine-tuning, defines model architectures
 - analyze: scripts defining analytic metrics and visualization
   - sum_of_squares: Sum of Squares metric calculation 
   - rsa: representative similarity anaylsis calculation
   - IDNN: methods for calculating Intrinsic Dimension from Alessio Ansuini
 - visualizations: create plots of results 
_______


### 0. Prerequisites
Change python path environment variable to local path, for packages to work.
```
export PYTHONPATH=/path/to/repo
```

_____
### 1. Training 

The files ``mnist_archs.py`` and ``vgg_arch.py` define the architectures used. 


#### 1.1 Pre-training CNN
Use example ``pretrain_mnist.sh`` shell script or command line tool.
This is how to specify the parameters for a ``train.py`` call

```
export CUDA_VISIBLE_DEVICES=0; 
python train.py --dataset 'mnist' --epochs 100 --bs 64 --lr 0.0001 --run_name 'pre_mnist' --seeds 10
```
same for all CNN training runs on MNIST-like datasets.


#### 1.2 Pre-training VGG
Use online available pre-trained models for VGG models (due to computational restrictions).
The file ``vgg16_pre.py`` loads, tests and saves a pre-trained model for each of the 6 datasets.

For the benchmarks used later the three target datasets get pre-trained like in the CNN case:
```
python train.py --dataset 'custom3D' --epochs 100 --bs 12 --lr 0.01 --run_name 'pre_custom3D' --seeds 10
python train.py --dataset 'Malaria' --epochs 100 --bs 22 --lr 0.01 --run_name 'pre_malaria' --seeds 10
python train.py --dataset 'Pets' --epochs 100 --bs 58 --lr 0.01 --run_name 'pre_pets' --seeds 10
```
learning rate starts with 0.01 because the lr_scheduler is used. 


#### 1.3 Fine-tuning
Same as before, use ``finetune.sh`` or call like
```
export CUDA_VISIBLE_DEVICES=0; 
python finetune.py --dataset 'mnist' --epochs 100 --bs 64 --lr 0.0001 --run_name 'ft_fashionmnist_mnist' --pre_dataset 'fashionmnist'
```
Same for ``finetune_vgg.py`` or ``finetune_vgg.sh``.


_____
_____

### 2. Analyze 

#### 2.1 Extract Activations
First extract the activation for samples of the target dataset on each pre-trained model with ``extract.py``by using ``extract.sh`` or the command line tool with
```
export CUDA_VISIBLE_DEVICES=0; 
python extract.py --trained_on 'mnist' --dataset 'fashionmnist' --model_folder 'all'
```

#### 2.2 Calculate Metrics 
Run ``analyze.py`` to calculate all three metrics and save to dataframe.
It uses the methods found in the files in the ``/metrics`` folder. 

_____
_____

### 3. Visualize

Plot accuracy results for CNN with ``plot_CNN.py`` and pre-training with ``plot_pretrain.py``.
``plot_vgg16.py`` plots the results for experiments on VGG architecture. 

Plot all metrics with ``plot_metrics.py`` by loading the analyze results from dataframes. 

The ``rsa.py`` file in ``/analyze/metrics `` also provides many visualizations for the exploration of different RSA results.
