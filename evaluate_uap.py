'''Code to evaluate the UAP for any target CNN.
   Almost similar to the code to train generator i.e. train_generator.py 
   except the optimization step; this code does only the forward pass
'''

from misc.layers import *
from classify import *
from new_generator import *
import tensorflow as tf
import numpy as np
import argparse
import h5py
import math
import random
import sys
import copy
from skimage.io import imread
from skimage.transform import resize
import time
import os
from random import shuffle
from math import ceil
from scipy import misc
from itertools import combinations
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

size=227
batch_size = 64

test_path = 'ilsvrc_valid.hdf5'
test_file = h5py.File(test_path, "r")
test_num = test_file["valid_img"].shape[0]
print "{} {}".format("Should be 1000",test_num)

def resize_caffenet(inp):

    for i in xrange(inp.shape[0]):

          img = resize(np.uint8(inp[i]) , (227,227))*255.0
          img = img.reshape((1,227,227,3))
          if i==0:
             out = img
          else:
             out = np.vstack([out,img])
    return out


def add_perturbation(inp, perturbation):
 
   inp = inp + perturbation 
   return inp      

def log_loss(prob_vec, adv_prob_vec, top_prob):
      
      size = prob_vec.get_shape().as_list()[0]
      for i in xrange(size):

               if i==0:
                    loss= adv_prob_vec[i][top_prob[i][0]]
               else:
                    loss = loss + adv_prob_vec[i][top_prob[i][0]]
 
      mean = (loss/size)
      gen_loss = -tf.log(1-mean)
      return gen_loss,mean

def cosine_loss(prob_vec, adv_prob_vec, top_prob):
      
      size = prob_vec.get_shape().as_list()[0]
      for i in xrange(size):
              a1 = tf.square(prob_vec[i])
              sum1 = tf.reduce_sum(a1)
              sqrt = tf.sqrt(sum1)
              unitv1 = tf.divide(a1,sqrt)
              a1 = tf.square(adv_prob_vec[i])
              sum1 = tf.reduce_sum(a1)
              sqrt = tf.sqrt(sum1)
              unitv2 = tf.divide(a1,sqrt)
              mul = tf.multiply(unitv1,unitv2)
              error = tf.reduce_sum(mul)

              if i==0:
                    loss= error#v_adv[i][topk[i][0]]
              else:
                    loss = loss + error#v_adv[i][topk[i][0]]

      cosine_loss = (loss/size)
     
      return cosine_loss

def validation_results(prob_adv,prob_real):
  nfool=0
  size = prob_real.shape[0]
  for i in xrange(size):
      if prob_real[i]!=prob_adv[i]:
         nfool = nfool+1
  return nfool, 100*float(nfool)/size      

def tensor_read(sess,img_list):
    filename_queue = tf.train.string_input_producer(img_list,num_epochs=2,shuffle=True)

    image_reader = tf.WholeFileReader()

    _, image_file = image_reader.read(filename_queue)

    image = tf.image.decode_png(image_file)

    image = tf.image.resize_images(image, [224, 224])
    image.set_shape((224, 224, 3))
    batch_size = 64
    num_preprocess_threads = 1
    min_queue_examples = 0
    image = tf.train.shuffle_batch(
    [image],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    capacity=min_queue_examples + 3 * batch_size,
    min_after_dequeue=min_queue_examples)



    # Required to get the filename matching to run.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())



    # Coordinate the loading of image files.

    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(coord=coord)
    #return sess.run(image) 
    image_tensor = sess.run(image)

    coord.request_stop()

    coord.join(threads)
    return image_tensor
                            

def read_images_as_array(img_list):
   
   for i,name in enumerate(img_list):
       
       img = imread(name)
       img = resize(img,(size,size))*255.0
       img = img.reshape((1,size,size,3))
#       print i,name,np.mean(img)
       if i==0:
         cln_img = img
         #adv_img = adv_im
       else:
         cln_img = np.vstack([cln_img,img])
         #adv_img = np.vstack(adv_img,adv_im)
   return cln_img#, adv_img

def tensor_preprocess(tensor, size=size):
    mean = [103.939, 116.779, 123.68]
    a = tensor[:,:,0]   
    a -= mean[2]
    a = tf.reshape(a,shape=(size,size,1))
    b = tensor[:,:,1]
    b -= mean[1]
    b = tf.reshape(b,shape=(size,size,1))
    c = tensor[:,:,2]
    c -= mean[0]
    c = tf.reshape(c,shape=(size,size,1))
    tensor = tf.concat([c,b,a],axis=2)
    #tensor[:,:,[0,1,2]] = tensor[:,:,[2,1,0]]
    return tensor

def whole_process(tensor):
    l = tensor.get_shape().as_list()[0]
    for i in xrange(l):
        if i==0:
            temp = tensor_preprocess(tensor[i])
            temp = tf.reshape(temp, shape = [1,size,size,3])
        else:   
            temp = tf.concat([temp, tf.reshape(tensor_preprocess(tensor[i]),shape=[1,size,size,3])],axis=0)
    return temp           



sess = tf.Session(config=config)
#sess2 = tf.Session(config=config)


img_dir = '/data1/nithish/WORK/Resources/ilsvrc_train/'
curr_dir = os.getcwd()
with sess.as_default():
 
 G = Generator()
 w = G.generate()
 w = tf.pad(w , [[0,0],[1,2],[1,2],[0,0]] , mode='REFLECT')
 saver = tf.train.Saver(tf.trainable_variables())
 img_list = open('text_files/new_train.txt').readlines()
 for i in xrange(len(img_list)):
   img_list[i] = img_list[i].split()[0]
 random.shuffle(img_list)  
 valid_list_og = open('text_files/new_valid.txt').readlines()
 for i in xrange(len(valid_list_og)):
   valid_list_og[i] = valid_list_og[i].split()[0]

 valid_list = open('text_files/full_valid.txt').readlines()
 for i in xrange(len(valid_list)):
   valid_list[i] = valid_list[i].split()[0]
 
 batch_data = tf.placeholder(dtype=tf.float32, shape=(batch_size,size,size,3), name='clean_train_data')
 val_data = tf.placeholder(dtype=tf.float32, shape=(batch_size,size,size,3), name='clean_validation_data') 
 class_hist = np.zeros([1000], dtype = np.int32) 

 random_adv_batch = add_perturbation(batch_data,w[0])
 adv_valid_data = add_perturbation(val_data,w[0])
 v,topk = scores(whole_process(batch_data))
 v_adv, topk_adv = scores(whole_process(random_adv_batch))         
 v  = v['prob']
 v_adv = v_adv['prob']
 q1_loss,mean_q1 = log_loss(v,v_adv,topk)
 cos_loss = cosine_loss(v,v_adv,topk)

 q1_loss = q1_loss
 cos_loss = cos_loss 
 lr = 1e-3
 train_op = tf.train.AdamOptimizer(lr).minimize(q1_loss)
 
 epoch=0 
 total_epochs = 60
 iterations = len(img_list) / batch_size
 remaining = len(img_list) % batch_size
 val_iterations = 1000/batch_size
 
 sess.run(tf.global_variables_initializer())
 
 G_list = ['cosine10','res4f_10','res5c_10']
 for i in G_list:
   model = 'variations/resnet50_'+i+'.00280.ckpt'
   saver.restore(sess,model)
   print "{} {}".format(i,"Loaded")
   for j in xrange(100):
 	
     	    z = np.random.uniform(low=-1. , high=1. , size = (G.batch_size,G.z_dim))
            print "{} {}".format(j+1,"z sampled")  

	    if  True:
		
		   total_fool=0 
		   print "{}".format("############### VALIDATION PHASE STARTED ################")

		   for y in xrange(val_iterations):

			i_s = y * batch_size  # index of the first image in this batch                  
			i_e = min([(y + 1) * batch_size, test_num])
			train_batch = test_file["test_img"][i_s:i_e, ...]
			train_batch = resize_caffenet(train_batch)               
			cln_top, adv_top = sess.run([topk, topk_adv],feed_dict={
			                                  G.zn:z,#np.random.uniform(low=-1. , high=1. , size = (G.batch_size,G.z_dim)),
			                                  batch_data:train_batch} )
	
			adv_top = np.reshape(adv_top , (adv_top.shape[0]))
		       
			for i in xrange(adv_top.shape[0]):
	   			class_hist[adv_top[i]] = class_hist[adv_top[i]] + 1		
			nfool,foolr = validation_results(cln_top,adv_top)
			total_fool = total_fool + nfool
			
			print "{} {} {} {} {} {} {} {} {}".format(y+1,"out of",val_iterations,"batches done and number of fools for this batch is",nfool,"/",batch_size,"and fooling rate",100*float(total_fool)/((y+1)*batch_size))
		  

		   foolr = 100*float(total_fool)/(val_iterations*batch_size)       
		   print "{} {} {}".format("Fooling rate",foolr,total_fool) 
		   np.save('cvpr_final/ablation_exp/'+i+'/'+str(j+1)+'.npy',class_hist)
			 	 
		   print "{}".format("############### VALIDATION PHASE ENDED ################")
	          
	  epoch = epoch + 1


 print "Ablation experiment Done!" 

        
 
