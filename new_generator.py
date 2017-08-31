import tensorflow as tf
from ops import linear, deconv2d
import numpy as np

class VBN(object):
    """
    Virtual Batch Normalization
    """

    def __init__(self, x, name, epsilon=1e-5, half=None):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        self.half = half
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(name) as scope:
            assert name.startswith("d_") or name.startswith("g_")
            self.epsilon = epsilon
            self.name = name
            if self.half is None:
                half = x
            elif self.half == 1:
                half = tf.slice(x, [0, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])
            elif self.half == 2:
                half = tf.slice(x, [shape[0] // 2, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])
            else:
                assert False
            self.mean = tf.reduce_mean(half, [0, 1, 2], keep_dims=True)
            self.mean_sq = tf.reduce_mean(tf.square(half), [0, 1, 2], keep_dims=True)
            self.batch_size = int(half.get_shape()[0])
            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None
            out = self._normalize(x, self.mean, self.mean_sq, "reference")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            self.reference_output = out

    def __call__(self, x):

        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        with tf.variable_scope(self.name) as scope:
            new_coeff = 1. / (self.batch_size + 1.)
            old_coeff = 1. - new_coeff
            new_mean = tf.reduce_mean(x, [1, 2], keep_dims=True)
            new_mean_sq = tf.reduce_mean(tf.square(x), [1, 2], keep_dims=True)
            mean = new_coeff * new_mean + old_coeff * self.mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
            out = self._normalize(x, mean, mean_sq, "live")
            if needs_reshape:
                out = tf.reshape(out, orig_shape)
            return out

    def _normalize(self, x, mean, mean_sq, message):
        # make sure this is called with a variable scope
        shape = x.get_shape().as_list()
        assert len(shape) == 4
        self.gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
        gamma = tf.reshape(self.gamma, [1, 1, 1, -1])
        self.beta = tf.get_variable("beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))
        beta = tf.reshape(self.beta, [1, 1, 1, -1])
        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None
        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
        out = x - mean
        out = out / std
        # out = tf.Print(out, [tf.reduce_mean(out, [0, 1, 2]),
        #    tf.reduce_mean(tf.square(out - tf.reduce_mean(out, [0, 1, 2], keep_dims=True)), [0, 1, 2])],
        #    message, first_n=-1)
        out = out * gamma
        out = out + beta
        return out


class Generator(object):
    def __init__(self, image_size=108, is_crop=True,
                 batch_size=32, image_shape=[128, 128, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 d_label_smooth=.25,
                 generator_target_prob=1.,
                 checkpoint_dir=None, sample_dir='samples',
                 generator=None,
                 generator_func=None, train=None, train_func=None,
                 generator_cls = None,
                 discriminator_func=None,
                 encoder_func=None,
                 build_model=None,
                 build_model_func=None, config=None,
                 devices=None,
                 disable_vbn=False,
                 sample_size=64,
		 out_init_b=0.,
                 out_stddev=.15):
                 
                 self.batch_size = batch_size
		 self.z_dim = z_dim
		 self.gf_dim = gf_dim
		 self.df_dim = df_dim
		 self.out_init_b = out_init_b
		 self.out_stddev = out_stddev 	
		 self.disable_vbn = disable_vbn 
		 self.y_dim = y_dim	
		 self.image_shape = image_shape	
                 self.zn = tf.placeholder(tf.float32, shape=[self.batch_size,self.z_dim]) 
    def vbn(self, tensor, name, half=None):
        if self.disable_vbn:
            class Dummy(object):
                def __init__(self, tensor, ignored, half):
                    self.reference_output=tensor
                def __call__(self, x):
                    return x
            VBN_cls = Dummy
        else:
            VBN_cls = VBN
        if not hasattr(self, name):
            vbn = VBN_cls(tensor, name, half=half)
            setattr(self, name, vbn)
            return vbn.reference_output
        vbn = getattr(self, name)
        return vbn(tensor)

    def generate(self, is_ref=False):
        """
        Builds the graph propagating from z to x.
        On the first pass, should make variables.
        All variables with names beginning with "g_" will be used for the
        generator network.
        """

        def make_z(shape, minval, maxval, name, dtype):
            assert dtype is tf.float32
            if is_ref:
                with tf.variable_scope(name) as scope:
                    z = tf.get_variable("z", shape,
                                initializer=tf.random_uniform_initializer(minval, maxval),
                                trainable=False)
                    if z.device != "/device:GPU:0":
                        print "z.device is " + str(z.device)
                        assert False
            else:
                z = tf.random_uniform(shape,
                                   minval=minval, maxval=maxval,
                                   name=name, dtype=tf.float32)
            return z


        z = make_z([self.batch_size, self.z_dim],
                                   minval=-1., maxval=1.,
                                   name='z', dtype=tf.float32)
        zs = [z]
        hlist = []
        hlist.append(z)
        make_vars = True


        def reuse_wrapper(packed, *args):
            """
            A wrapper that processes the output of TensorFlow calls differently
            based on whether we are reusing Variables or not.
            Parameters
            ----------
            packed: The output of the TensorFlow call
            args: List of names
            If make_vars is True, then `packed` will contain all the new Variables,
            and we need to assign them to self.foo fields.
            If make_vars is False, then `packed` is just the output tensor, and we
            just return that.
            """
            if make_vars:
                assert len(packed) == len(args) + 1, len(packed)
                out = packed[0]
            else:
                out = packed
            return out

        assert not self.y_dim
        # project `z` and reshape
        z_ = reuse_wrapper(linear(self.zn, self.gf_dim*7*4*4, 'g_h0_lin', with_w=make_vars), 'h0_w', 'h0_b')

        h0 = tf.reshape(z_, [-1, 4, 4, self.gf_dim * 7])
        hlist.append(h0)
        h0 = tf.nn.relu(self.vbn(h0, "g_vbn_0"))
        h0z = make_z([self.batch_size, 4, 4, self.gf_dim],
                                   minval=-1., maxval=1.,
                                   name='h0z', dtype=tf.float32)
        zs.append(h0z)
        
        h0 = tf.concat([h0, h0z],3)
        
        hlist.append(h0) 
        h1 = reuse_wrapper(deconv2d(h0,
            [self.batch_size, 7, 7, self.gf_dim*4], name='g_h1', with_w=make_vars),
            'h1_w', 'h1_b')
        h1 = tf.nn.relu(self.vbn(h1, "g_vbn_1"))
        h1z = make_z([self.batch_size, 7, 7, self.gf_dim],
                                   minval=-1., maxval=1.,
                                   name='h1z', dtype=tf.float32)
        zs.append(h1z)
        h1 = tf.concat([h1, h1z],3)
        hlist.append(h1)
        h2 = reuse_wrapper(deconv2d(h1,
            [self.batch_size, 14, 14, self.gf_dim*2], name='g_h2', with_w=make_vars),
            'h2_w', 'h2_b')
        h2 = tf.nn.relu(self.vbn(h2, "g_vbn_2"))
        half = self.gf_dim // 2
        if half == 0:
            half = 1
        h2z = make_z([self.batch_size, 14, 14, half],
                                   minval=-1., maxval=1.,
                                   name='h2z', dtype=tf.float32)
        zs.append(h2z)
        h2 = tf.concat([h2, h2z],3)
        hlist.append(h2)

        i1 = tf.random_uniform(shape=[], dtype=tf.int32,minval=0, maxval=self.batch_size)
        i2 = tf.random_uniform(shape=[], dtype=tf.int32,minval=0, maxval=self.batch_size)
        feature_loss = -tf.reduce_mean(tf.squared_difference(h2[i1],h2[i2]))

        h3 = reuse_wrapper(deconv2d(h2,
            [self.batch_size, 28, 28, self.gf_dim*1], name='g_h3', with_w=make_vars),
            'h3_w', 'h3_b')
        if make_vars:
            h3_name = "h3_relu_first"
        else:
            h3_name = "h3_relu_reuse"
        h3 = tf.nn.relu(self.vbn(h3, "g_vbn_3"), name=h3_name)
        

        quarter = self.gf_dim // 4
        if quarter == 0:
            quarter = 1
        h3z = make_z([self.batch_size, 28, 28, quarter],
                                   minval=-1., maxval=1.,
                                   name='h3z', dtype=tf.float32)
        zs.append(h3z)
        h3 = tf.concat([h3, h3z],3)
        hlist.append(h3)
        assert self.image_shape[0] == 128

        h4 = reuse_wrapper(deconv2d(h3,
                [self.batch_size, 56, 56, self.gf_dim*1],
                name='g_h4', with_w=make_vars),
            'h4_w', 'h4_b')
        h4 = tf.nn.relu(self.vbn(h4, "g_vbn_4"))
      

        eighth = self.gf_dim // 8
        if eighth == 0:
            eighth = 1
        h4z = make_z([self.batch_size, 56, 56, eighth],
                                   minval=-1., maxval=1.,
                                   name='h4z', dtype=tf.float32)
        zs.append(h4z)
        h4 = tf.concat([h4, h4z],3)
        hlist.append(h4)

        h5 = reuse_wrapper(deconv2d(h4,
                [self.batch_size, 112, 112, self.gf_dim * 1],
                name='g_h5', with_w=make_vars),
            'h5_w', 'h5_b')
        h5 = tf.nn.relu(self.vbn(h5, "g_vbn_5"))
      
      

        sixteenth = self.gf_dim // 16
        if sixteenth == 0:
            sixteenth = 1
        h5z = make_z([self.batch_size, 112, 112, eighth],
                                   minval=-1., maxval=1.,
                                   name='h5z', dtype=tf.float32)
        zs.append(h5z)
        h5 = tf.concat([h5, h5z],3)
	hlist.append(h5)
	

        h6 = reuse_wrapper(deconv2d(h5,
                [self.batch_size, 224, 224, self.gf_dim * 1],
                name='g_h6', with_w=make_vars),
            'h6_w', 'h6_b')
        h6 = tf.nn.relu(self.vbn(h6, "g_vbn_6"))
       

        sixteenth = self.gf_dim // 16
        if sixteenth == 0:
            sixteenth = 1
        h6z = make_z([self.batch_size, 224, 224, eighth],
                                   minval=-1., maxval=1.,
                                   name='h6z', dtype=tf.float32)
        zs.append(h6z)
        h6 = tf.concat([h6, h6z],3)

	hlist.append(h6)

        h7 = reuse_wrapper(deconv2d(h6,
                [self.batch_size, 224, 224, 3],
                d_w = 1, d_h = 1,
                name='g_h7', with_w=make_vars,
                init_bias=self.out_init_b,
                stddev=self.out_stddev),
            'h7_w', 'h7_b')
        hlist.append(h7)


        out = 10*tf.tanh(h7)
        hlist.append(out)
      
	return out#,zs,hlist

