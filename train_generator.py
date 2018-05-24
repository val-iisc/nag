from misc.layers import *
from new_generator import *
import tensorflow as tf
import numpy as np
import argparse
from classify import *
import random
import sys
import copy
from skimage.io import imread
from skimage.transform import resize
import time
import os
import h5py
from random import shuffle
from math import ceil
import scipy
import pdb

#config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0))
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

size=224
batch_size = 32

#Defining the hdf5 files for training and validation
train_path = 'ilsvrc_train.hdf5'
train_file = h5py.File(train_path, "r")
train_num = train_file["train_img"].shape[0]
valid_path = 'ilsvrc_valid.hdf5'
valid_file = h5py.File(valid_path, "r")
val_num = valid_file["valid_img"].shape[0]
batches_list = list(range(int(ceil(float(train_num) / batch_size))))


def add_perturbation(inp, perturbation):
   inp = inp + perturbation 
   return inp      

def add_perturbation2(inp,perturbation):
   ''' Helper function to add the perturbations in a random order (for the shuffled perturbed batch)
   '''   
   k = inp.get_shape().as_list()[0]
   for i in xrange(k):
   	j = tf.random_uniform(shape=[], dtype=tf.int32,minval=0, maxval=batch_size)
        temp = tf.reshape((inp[i] + perturbation[j]),shape = [1,size,size,3])
        if i==0:
           out = temp
        else:
           out = tf.concat([out,temp],axis=0)
   return out 
   
def log_loss(prob_vec, adv_prob_vec, top_prob):
      '''Helper function to computer compute -log(1-qc'), 
       where qc' is the adversarial probability of the class having 
       maximum probability in the corresponding clean probability
       Parameters: 
       prob_vec : Probability vector for the clean batch
       adv_prob_vec : Probability vecotr of the adversarial batch
       Returns: 
       -log(1-qc') , qc'
      '''  
      size = prob_vec.get_shape().as_list()[0]
      for i in xrange(size):

               if i==0:
                    loss= adv_prob_vec[i][top_prob[i][0]]
               else:
                    loss = loss + adv_prob_vec[i][top_prob[i][0]]
 
      mean = (loss/size)
      gen_loss = -tf.log(1-mean)
      return gen_loss,mean

def cosine_loss(prob_vec, adv_prob_vec):
      '''Helper function to calculate the cosine distance between two probability vectors
         Parameters: 
         prob_vec : Probability vector for the clean batch
         adv_prob_vec : Probability vector for the adversarial batch
         Returns : 
         Cosine distance between the corresponding clean and adversarial batches
      '''    
      size = prob_vec.get_shape().as_list()[0]
      for i in xrange(size):
              normalize_a = tf.nn.l2_normalize(prob_vec[i],0)        
	      normalize_b = tf.nn.l2_normalize(adv_prob_vec[i],0)
	      cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b))

              if i==0:
                    loss= cos_similarity
              else:
                    loss = loss + cos_similarity

      cosine_loss = (loss/size)
     
      return cosine_loss

def feature_distance(fa,fb):
     '''Function to maximize the L2 distance between two feature maps
        Parameters:
        fa: first feature map
        fb: second feature map
        Returns:
        Mean L2 difference between provided feature maps
     '''
     size = fa.get_shape.as_list()[0]
     for i in xrange(size):
         temp = -tf.reduce_mean(tf.squared_difference(fa[i],fb[i]))
         if i==0:
            loss= temp
         else:
            loss = loss + temp
     loss = loss/size
     return loss

def validation_results(prob_adv,prob_real):
  '''Helper function to calculate mismatches in the top index vector
     for clean and adversarial batch
     Parameters:
     prob_adv : Index vector for adversarial batch
     prob_real : Index vector for clean batch
     Returns:
     Number of mismatch and its percentage
  '''
    
  nfool=0
  size = prob_real.shape[0]
  for i in xrange(size):
      if prob_real[i]!=prob_adv[i]:
         nfool = nfool+1
  return nfool, 100*float(nfool)/size      
                            

def read_images_as_array(img_list):
   ''' Helper function to read images as numpy arrays from paths
       NAIVE WAY: Not using it right now
       Parameters
       img_list : list containing the paths of the images to be read
       Returns
       numpy array of shape [length of list, 224,224,3]
    '''     
   for i,name in enumerate(img_list):
       print "{} {}".format("hiiii",i)
       img = imread(name)
       img = resize(img,(size,size))*255.0
       img = img.reshape((1,size,size,3))
       if i==0:
         cln_img = img
         
       else:
         cln_img = np.vstack([cln_img,img])
         
   return cln_img




def resize_caffenet(inp):
    '''Utility function to resize the batch of images 
       to [227,227,3] for CaffeNet
       Parameters:
       Input batch
       Returns:
       Resized batch
    '''
       
 
    for i in xrange(inp.shape[0]):
            
          img = resize(np.uint8(inp[i]) , (227,227))*255.0
	  img = img.reshape((1,227,227,3))
          if i==0:
             out = img
          else: 
 	     out = np.vstack([out,img])
    return out  	
            

'''Helper functions to process the tensors before feeding them 
   to the target CNNs
'''
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




img_dir = '/data1/nithish/WORK/Resources/ilsvrc_train/'
curr_dir = os.getcwd()

#Instantiating the generator object and the perturbation
G = Generator()
w = G.generate()
saver = tf.train.Saver(tf.trainable_variables()) 
print "{} {}".format("Shape of the perturbation",w.get_shape().as_list())

batch_data = tf.placeholder(dtype=tf.float32, shape=(batch_size,size,size,3), name='clean_train_data') 
	
random_adv_batch = add_perturbation(batch_data,w)
random_adv_batch2 = add_perturbation2(batch_data,w) # Adding the same perturbation but in different (random) order than in 'random_adv_batch'
assert random_adv_batch.get_shape().as_list() == random_adv_batch2.get_shape().as_list()

# v,v_adv and v_adv2 are the dictionary containing activation of the layers of target CNN; topk and topk_adv are the indices of the class with highest probability 
v,topk = scores(whole_process(batch_data))  
v_adv, topk_adv = scores(whole_process(random_adv_batch))         
v_adv2,_ = scores(whole_process(random_adv_batch2))

''' Following two statements extract the activations of any intermediate layer for:
    1. batch of images (of batch size=32) corrupted by the perturbations (of batch size=32) 
    2. same batch of images corrupted by same batch of perturbations but in different (random) order
    (in this case the intermdeiate layer is set to 'res4f' of ResNet 50 architecture)
'''
f1_res4f = v_adv['res4f']
f2_res4f = v_adv2['res4f']

#Extracting the probability vectors for clean batch, perturbed batch, differently perturbed batch
v  = v['prob']
v_adv = v_adv['prob']
v_adv2 = v_adv2['prob']
feature_loss = -10*tf.reduce_mean(tf.squared_difference(f1_res4f,f2_res4f))#feature_distance(f1_res4f,f2_res4f)
q1_loss,mean_q1 = log_loss(v,v_adv,topk)
cos_loss = cosine_loss(v,v_adv)

q1_loss = q1_loss + feature_loss 
cos_loss = cos_loss + feature_loss
lr = 1e-3
train_op = tf.train.AdamOptimizer(lr).minimize(q1_loss)

epoch=0 
total_epochs = 60
iterations = train_num/batch_size
remaining = len(img_list) % batch_size
val_iterations = val_num/batch_size

with tf.Session(config=config) as sess:
 sess.run(tf.global_variables_initializer())
 dt = train_file["train_img"][:, ...]
 while  epoch < total_epochs:
  #Shuffling the data after every epoch
  np.random.shuffle(dt)
  print "{} {} {}".format("**********************",epoch,"************************")
  for y in xrange(iterations):
      
    i_s = y * batch_size                    
    i_e = min([(y + 1) * batch_size, train_num]) 
    
    a = time.time()  
    
    train_batch = dt[i_s:i_e, ...] # Sample a batch from the training data                               
    b = time.time()
 
    # Optimization step
    _,backprop_loss,f_loss,meant = sess.run([train_op,q1_loss,feature_loss,mean_q1], feed_dict={
                                G.zn:np.random.uniform(low=-1. , high=1. ,size = (G.batch_size,G.z_dim)),
                                batch_data:train_batch})   
    c = time.time()

    print "{} {} {} {}".format("Time for reading the data"(b-a),"Time for optimization",(c-b))
    
    # Comment out the below statements if the evolving perturbations don't have to be monitored
    if y%30==0:
       np.save('running_perturbation.npy',sess.run(w, feed_dict={G.zn:np.random.uniform(low=-1. , high=1. ,size = (G.batch_size,G.z_dim))}))                                           
    
    print "{} {} {} {} {} {} {} {} {} {}".format("Epoch",epoch,"Iteration",y,"Log loss",backprop_loss,"Mean",meant,"Feature loss",f_loss)
    print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    f = open('log_loss_imagenet.txt','a')
    f.write(str(backprop_loss)+'\n')
    f.close()


    # Saving the model as a checkpoint file at the end of each epoch
    if  y==(iterations-1):    
   
            saver.save(sess,"log{}.{:05d}.ckpt".format(epoch,y))



    if  y!=0 and y%100==0:
           total_fool=0 
           print "{}".format("############### VALIDATION PHASE STARTED ################")

           for y in xrange(val_iterations):

                
                i_s = y * batch_size  # index of the first image in this batch                  
                i_e = min([(y + 1) * batch_size, val_num])
                train_batch = valid_file["valid_img"][i_s:i_e, ...]
                
                cln_top, adv_top = sess.run([topk, topk_adv],feed_dict={
                                                  G.zn:np.random.uniform(low=-1., high=1. , size = (G.batch_size,G.z_dim)),
                                                  batch_data:train_batch} )
                nfool,foolr = validation_results(cln_top,adv_top)
                total_fool = total_fool + nfool
                print "{} {} {} {}".format(y+1,"out of",val_iterations,"done")


           foolr = 100*float(total_fool)/(val_iterations*batch_size)       
           print "{} {} {}".format("Fooling rate",foolr,total_fool) 
           f = open('log_fool_rate_imagenet.txt','a')
           f.write(str(foolr)+'\n')
           f.close()

                 
         
           print "{}".format("############### VALIDATION PHASE ENDED ################")
           
  epoch = epoch + 1

  
 
