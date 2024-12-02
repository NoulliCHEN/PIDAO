# PIDAO

### This repository contains all the codes for our paper published in Nature Conmmunications:

[Chen, S., Liu, J., Wang, P., Xu, C., Cai, S. & Chu, J. Accelerated optimization in deep learning with a proportional-integral-derivative controller. Nat Commun 15, 10263 (2024).](https://doi.org/10.1038/s41467-024-54451-3)

This repository is provided for research purposes only. 

## Requirements

- CUDA 10.2
- Python 3.8
- pytorch 1.10.2
- numpy
- scipy
- tensorboard
- torchvision 0.11.3
- torchaudio 0.10.2
- os
- matplotlib
- pyDOE
- h5py

## Installation

```shell
conda create -n PIDAO python=3.8
conda activate PIDAO
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=10.2 -c pytorch -c conda-forge
conda install tensorboard scipy os matplotlib pyDOE h5py
```

It can be installed within half an hour, mainly depending on the internet speed.

## Required Data and Demo

We provide a detailed readme file to demo the software/code in each case's folder. The download link of each required dataset is also provided in the readme file.

- data_classfication folder >> MNIST/FashionMNIST/Cifar10 classification
- MNIST_training_path_projection folder >> the projection of the training path of MNIST classification
- Continuous_time_comparison >> the comparison of different continuous-time optimizers
- Discrete_time_comparison >> the comparison of different discrete-time optimizers
- FNO1d_Burgers >> FNO for learning Burgers' equation
- FNO2d_darcy >> FNO for learning Darcy flow
- PINNs_Burgers >> PINNs for learning Burgers' equation
- PINNs_cavity >> PINNs for learning cavity flow
- ModelNet40 >> point cloud classification

If you use the codes in your research work, please cite the following paper: 
  
	@article{chen2024accelerated,
  	  title={Accelerated optimization in deep learning with a proportional-integral-derivative controller},
  	  author={Chen, Song and Liu, Jiaxu and Wang, Pengkai and Cai, Shengze and Xu, Chao and Chu, Jian},
  	  journal={Nature Communications},
  	  volume={15},
          number={10263},
  	  year={2024}
	}

