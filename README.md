# Modelling universal adversarial perturbations
(Work in progress)
## Introduction
This repository contains the code for learning a manifold of perturbations that can easiy fool current state-of-the-art CNNs, using a generative model. At present, this repo provides the facility to train the generator that can produce perturbations to fool VGG F, VGG 16, VGG 19, GoogleNet, CaffeNet, ResNet 50, ResNet 152.

## Architecture
![](/extras/architecture.png)

## Dependencies
```
Python 2.7.1, Tensorflow 1.0.1
```
## Sample perturbations
![](/extras/perturbations.png)

## Generalizability of universal adversarial perturbations
The table below shows the fooling rate achieved for different networks. The rows represent the network for which the perturbation is crafted, and the column indicates the netwrok on which the strength of perturbation is tested. Testing is done on the 50k validation set of ILSVRC.  

![](/extras/fr_table.png)

(more to follow...)




