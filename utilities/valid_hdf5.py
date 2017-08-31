from random import shuffle
import glob
import sys
hdf5_path = 'ilsvrc_valid.hdf5'
img_path = '/data1/nithish/WORK/Resources/ilsvrc_valid/*.png'
addrs = glob.glob(img_path)

train_addrs = addrs[0:int(1.0*len(addrs))]
#valid_list = open('full_valid.txt').readlines()
#for i in xrange(len(valid_list)):
#   valid_list[i] = valid_list[i].split()[0]
#print "{} {}".format("Length of train list",len(valid_list))



import numpy as np
import h5py
from skimage.io import imread
from skimage.transform import resize
size=224

train_shape = (len(train_addrs), 224, 224, 3)
#val_shape = (len(val_addrs), 224, 224, 3)
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("valid_img", train_shape, np.float32)
#hdf5_file.create_dataset("val_img", val_shape, np.int8)
#hdf5_file.create_dataset("test_img", test_shape, np.int8)

for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 100 == 0 and i > 1:
        print 'Valid data: {}/{}'.format(i, len(train_addrs))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    
    img = imread(addr)
   
    img = resize(img, (224, 224))*255.0
    img2 = img
    img = img.reshape((1,size,size,3))
 
    hdf5_file["valid_img"][i, ...] = img2
 


#for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
#    if i % 100 == 0 and i > 1:
#        print 'Valid data: {}/{}'.format(i, len(val_addrs))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
#    addr = val_addrs[i]
#    img = imread(addr)
#    img = resize(img, (224, 224))*255.0
#    img = img.reshape((1,size,size,3))
#    hdf5_file["val_img"][i, ...] = img[None]


hdf5_file.close()

