import numpy as np
import scipy.misc
from skimage.io import imread
from skimage.transform import resize
import os

fetch_dir = "/data1/nithish/WORK/Resources/new_train_data/"
append_dir = "/data1/nithish/WORK/Resources/ilsvrc_train/"
#os.chdir(fetch_dir)


valid_list = open('append_train.txt').readlines()
for i in xrange(len(valid_list)):
  valid_list[i] = valid_list[i].split()[0]

for i,name in enumerate(valid_list):
   os.chdir(fetch_dir)
   im = imread(valid_list[i])
   if len(im.shape)==2:
      im = np.dstack([im,im,im])
   l,r = im.shape[0], im.shape[1]
   lnew = (l-256)/2
   rnew = (r-256)/2
   inew = im[lnew:lnew+256,rnew:rnew+256,:]
   os.chdir(append_dir)
   scipy.misc.imsave(valid_list[i][:-5]+'.png',inew)
   print "{} {}".format(i,"Done")
