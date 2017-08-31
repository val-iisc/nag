import h5py
import numpy as np
hdf5_path = 'temp_5.hdf5'
subtract_mean = False
    # open the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")
    # subtract the training mean
if subtract_mean:
        mm = hdf5_file["train_mean"][0, ...]
        mm = mm[np.newaxis, ...]
    # Total number of samples
data_num = hdf5_file["train_img"].shape[0]
val = hdf5_file["val_img"].shape[0]  
print data_num,val


from random import shuffle
from math import ceil
import matplotlib.pyplot as plt
    # create list of batches to shuffle the data
batch_size=64
batches_list = list(range(int(ceil(float(data_num) / batch_size))))
shuffle(batches_list)
    # loop over batches
for n, i in enumerate(batches_list):
        i_s = i * batch_size  # index of the first image in this batch
        i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch
        # read batch images and remove training mean
        images = hdf5_file["train_img"][i_s:i_e, ...]
        print images.shape
        # read labels and convert to one hot encoding
        #labels = hdf5_file["train_labels"][i_s:i_e]
        #labels_one_hot = np.zeros((batch_size, nb_class))
        #labels_one_hot[np.arange(batch_size), labels] = 1
        #print labels[0]#, labels_one_hot[0, :]
        #plt.imshow(images[0])
        #plt.show()
        if n == 5:  # break after 5 batches
            break
hdf5_file.close()
