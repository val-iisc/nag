# NAG - Network for Adversary Generation


Code for the paper **[NAG - Network for Adversary Generation](https://arxiv.org/abs/1712.03390)**. 

Konda Reddy Mopuri*, Utkarsh Ojha*, Utsav Garg, R. Venkatesh Babu.

**[CVPR 2018](cvpr2018.thecvf.com)**

This work is an attempt to explore the manifold of perturbations that can cause CNN based classifiers to behave absurdly. At present, this repository provides the facility to train the generator that can produce perturbations to fool VGG F, VGG 16, VGG 19, GoogleNet, CaffeNet, ResNet 50, ResNet 152. The generator architecture has been modified from **[here](https://github.com/openai/improved-gan/tree/master/imagenet)**.

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
## Setting up the dataset

Our major results have been obtained by training our model on a dataset consisting of 10k images, 10 random images from each of the 1000 classes from ILSVRC dataset. Testing has been done on the standard 50k validation images of ILSVRC. To speed up the training and testing of the model, data in .hdf5 format has been used. ```utilities/``` folder provides the relevant scripts to convert a folder of images in a suitable format (.jpg, .png etc.) into a single .hdf5 file. 

For example, ```train_hdf5.py``` does the necessary pre-processing on a folder of images and creates ```ilsvrc_train.hdf5``` file. 

**Caution:**  .hdf5 files created will be of large sizes. ```ilsvrc_test.hdf5``` will be ~30gb. 



## Training your own generator

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

