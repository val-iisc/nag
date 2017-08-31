import tensorflow as tf
from new_generator import *
import sys
import scipy

sess = tf.Session()    
sess.run(tf.global_variables_initializer())      

#Creating the generator object followed by perturbation (V is the perturbation of dimension [batch_size,224,224,3])
G = Generator()
V = G.generate()

#Declaring the variables to be loaded followed by restoring the generator model
saver = tf.train.Saver(tf.trainable_variables())
saver.restore(sess,'checkpoints10k/googlenet.00139.ckpt')

n=20 # 'n' denotes the number of perturbations to be generated 
for i in xrange(n): 
   D = sess.run(V, feed_dict={G.zn:np.random.uniform(low=-1.,high=1.,size = (G.batch_size,G.z_dim))})
   np.save('perturbation'+str(i)+'.npy',D[0]) # Comment out this line if perturbations are only needed as .png files  
   scipy.misc.imsave('perturbation'+str(i)+'.png',D[0]) #Comment out this line if perturbations are only needed as .npy files
  
print "{} {}".format(n,"perturbations saved")
