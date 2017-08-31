import tensorflow as tf
import numpy as np
import sys
def batchnormalize(X, eps=1e-8, g=None, b=None):
    return X
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1] )
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean))
        X = (X-mean) / tf.sqrt(std+eps)#std

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return -(t * tf.log(o) + (1.- t)*tf.log(1. - o))

class Generator():
    def __init__(
            self,
            batch_size=1,
            image_shape=[224,224,3],
            dim_z=100,
            dim_W1=1024,
            dim_W2=512,
            dim_W3=256,
            dim_W4=128,
            dim_W5=64,
            dim_W6=3
            ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_W4 = dim_W4
        self.dim_W5 = dim_W5
        self.dim_W6 = dim_W6
        self.gen_W1 = tf.Variable(tf.truncated_normal([dim_z, dim_W1*7*7], stddev=0.02), name='gen_W1')
        self.gen_bn_g1 = tf.Variable( tf.truncated_normal([dim_W1*7*7], mean=1.0, stddev=0.02), name='gen_bn_g1')
        self.gen_bn_b1 = tf.Variable( tf.zeros([dim_W1*7*7]), name='gen_bn_b1')

        self.gen_W2 = tf.Variable(tf.truncated_normal([5,5,dim_W2, dim_W1], stddev=0.02), name='gen_W2')
        self.gen_bn_g2 = tf.Variable( tf.truncated_normal([dim_W2], mean=1.0, stddev=0.02), name='gen_bn_g2')
        self.gen_bn_b2 = tf.Variable( tf.zeros([dim_W2]), name='gen_bn_b2')

        self.gen_W3 = tf.Variable(tf.truncated_normal([5,5,dim_W3, dim_W2], stddev=0.02), name='gen_W3')
        self.gen_bn_g3 = tf.Variable( tf.truncated_normal([dim_W3], mean=1.0, stddev=0.02), name='gen_bn_g3')
        self.gen_bn_b3 = tf.Variable( tf.zeros([dim_W3]), name='gen_bn_b3')

        self.gen_W4 = tf.Variable(tf.truncated_normal([5,5,dim_W4, dim_W3], stddev=0.02), name='gen_W4')
        self.gen_bn_g4 = tf.Variable( tf.truncated_normal([dim_W4], mean=1.0, stddev=0.02), name='gen_bn_g4')
        self.gen_bn_b4 = tf.Variable( tf.zeros([dim_W4]), name='gen_bn_b4')

        self.gen_W5 = tf.Variable(tf.truncated_normal([5,5,dim_W5, dim_W4], stddev=0.02), name='gen_W5')
        self.gen_bn_g5 = tf.Variable( tf.truncated_normal([dim_W5], mean=1.0, stddev=0.02), name='gen_bn_g5')
        self.gen_bn_b5 = tf.Variable( tf.zeros([dim_W5]), name='gen_bn_b5')
        
        self.gen_W6 = tf.Variable(tf.truncated_normal([5,5,dim_W6, dim_W5], stddev=0.02), name='gen_W5')
        

        self.gen_params = [
                self.gen_W1, self.gen_bn_g1, self.gen_bn_b1,
                self.gen_W2, self.gen_bn_g2, self.gen_bn_b2,
                self.gen_W3, self.gen_bn_g3, self.gen_bn_b3,
                self.gen_W4, self.gen_bn_g4, self.gen_bn_b4,
                self.gen_W5, self.gen_bn_g4, self.gen_bn_b4,
                self.gen_W6]
        self.Z = tf.placeholder(tf.float32, [self.batch_size,self.dim_z])
        self.saver = tf.train.Saver(self.gen_params, max_to_keep=100)
       
   

    def generate(self):
        h1 = lrelu(batchnormalize((tf.matmul(self.Z,self.gen_W1)), g=self.gen_bn_g1, b=self.gen_bn_b1))
        #return (tf.matmul(self.Z, self.gen_W1))#, g=self.gen_bn_g1, b=self.gen_bn_b1)
        h1 = tf.reshape(h1, [self.batch_size,7,7,self.dim_W1])

        output_shape_l2 = [self.batch_size,14,14,self.dim_W2]
        h2 = tf.nn.conv2d_transpose(h1, self.gen_W2, output_shape=output_shape_l2, strides=[1,2,2,1])
        h2 = lrelu( batchnormalize(h2, g=self.gen_bn_g2, b=self.gen_bn_b2) )
        output_shape_l3 = [self.batch_size,28,28,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = lrelu( batchnormalize(h3, g=self.gen_bn_g3, b=self.gen_bn_b3) )
        

        output_shape_l4 = [self.batch_size,56,56,self.dim_W4]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        h4 = lrelu( batchnormalize(h4, g=self.gen_bn_g4, b=self.gen_bn_b4) )
       # return 10.0*tf.tanh(h4)
        output_shape_l5 = [self.batch_size,112,112,self.dim_W5]
        h5 = tf.nn.conv2d_transpose(h4, self.gen_W5, output_shape=output_shape_l5, strides=[1,2,2,1])
        h5 = lrelu( batchnormalize(h5, g=self.gen_bn_g5, b=self.gen_bn_b5) )
       # return 10.0*tf.tanh(h5)
        output_shape_l6 = [self.batch_size,224,224,self.dim_W6]
        h4 = tf.nn.conv2d_transpose(h5, self.gen_W6, output_shape=output_shape_l6, strides=[1,2,2,1])
        #h4 = lrelu( batchnormalize(h4, g=self.gen_bn_g4, b=self.gen_bn_b4) )
        return 10.0*tf.tanh(0.5*h4)
        # h4 = tf.clip_by_value(h4,clip_value_min=-10, clip_value_max=10)
        # return h4
        

      #  x = 10.0*tf.tanh(h5)
       # return x

      #  x = 10.0*tf.tanh(h5)
       # return x

    def save_model(self,file_name):
        self.saver.save(self.sess,file_name)

    def load_model(self,file_name):
        self.saver.restore(self.sess, file_name)


    # def get_gradients(self,total_loss):
    #   grads = self.optimizer.compute_gradients(total_loss)

    #   return grads

    # def get_backprop(self,total_loss,global_step):
    #   solver = self.optimizer.minimize(total_loss,global_step=global_step)
    #   return solver
    


   


