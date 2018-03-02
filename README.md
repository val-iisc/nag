# NAG - Network for Adversary Generation

## Introduction
Code for the paper **[NAG - Network for Adversary Generation](https://arxiv.org/abs/1712.03390)**. 

Konda Reddy Mopuri*, Utkarsh Ojha*, Utsav Garg, R. Venkatesh Babu.

At present, this repo provides the facility to train the generator that can produce perturbations to fool VGG F, VGG 16, VGG 19, GoogleNet, CaffeNet, ResNet 50, ResNet 152. The generator architecture has been modified from **[here](https://github.com/openai/improved-gan/tree/master/imagenet)**.

## Architecture
![](/extras/nag.png)

## Sample perturbations
![](/extras/pb_nag.png)

## Generalizability of universal adversarial perturbations
The table below shows the fooling rate achieved for different networks. The rows represent the network for which the perturbation is crafted, and the column indicates the netwrok on which the strength of perturbation is tested. Testing is done on the 50k validation set of ILSVRC.  

![](/extras/nag_table.png)




## Dependencies
```
Python 2.7.1, Tensorflow 1.0.1, h5py
```
## Training your own generator

We've shown the results by training the generator on 10k images and evaluating it on ILSVRC 50k validation images. Accordingly, our training code requires existence of relevant .hdf5 files. To create these for training, validation and testing, the utilities/ folder provides the relevant scripts. For example, the ```train_hdf5.py ``` creates the ```ilsvrc_train.hdf5 ``` file which acts as the training data. The ```img_path``` provides the location where the actual training images are stored in usual form (ex .jpg). Repeat this for validation and testing set. As a caution, the .hdf5 file for testing data (containing 50k images) will occupy ~30 GB; ensure that sufficient space is available in the disk.

Once the .hdf5 files are done, we can begin training the generator for a given classifier. Set the target classifier in ```classify.py  ``` (currently the default network is ResNet 50) from any one of the mentioned networks.

For training the generator, run:
``` python train_generator.py ```
The code saves the evolving perturbations by saving them in ```running_perturbatin.npy ``` in case one wants to visualize them.

## Testing on clean images

Run ``` python uap_generate.py ``` to obtain the perturbations from the saved generator in the form of either .png images or .npy files. Simple add the perturbations to the clean images and test their classification results of clean and corrupted image using ```classify_image.py ```. As a word of caution, make sure you set the classifier in ```classify_image.py``` according to the loaded checkpoint file in ```uap_generate.py```, unless you want to check the transferability of perturbations.

## Sample fooling

![](/extras/example.png)

## Reference
```
@inproceedings{nag-cvpr-2018,
  title={NAG: Network for Adversary Generation},
  author={Mopuri, Konda Reddy and Ojha, Utkarsh and Garg, Utsav and Babu, R Venkatesh},
 booktitle = {Proceedings of the IEEE Computer Vision and Pattern Recognition ({CVPR})},
 year = {2018}
}
```

Contact **[Utkarsh Ojha](https://utkarshojha.github.io/)** in case you have any questions.

