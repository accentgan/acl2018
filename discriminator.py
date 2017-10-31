from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
from ops import *
import numpy as np


def discriminator(self, wave_in, zvalue, flag=True, reuse=True):
        """
        wave_in: waveform input
        """
        # take the waveform as input "activation"
        zdim = zvalue.get_shape().as_list()[-1]
        zstack = tf.reshape(zvalue,shape=[self.batch_size, 1, zdim])
        in_dims = wave_in.get_shape().as_list()
        hi = wave_in
        if len(in_dims) == 2:
            hi = tf.expand_dims(wave_in, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Discriminator input must be 2-D or 3-D')

        batch_size = int(wave_in.get_shape()[0])

        # set up the disc_block function
        if flag:
            fmaps = self.d_num_fmaps
            if hasattr(self,"d_spec_fmaps"):
                large = self.d_spec_fmaps
            else :
                large = []
            name_disc = "d_model"
        else :
            fmaps = self.d_large_num_fmaps
            large = self.d_large_spec_fmaps
            name_disc = "d_large_model"
        with tf.variable_scope(name_disc) as scope:
            if reuse:
                scope.reuse_variables()
            def disc_block(block_idx, input_, kwidth, nfmaps, bnorm, activation,
                           pooling=2):
                with tf.variable_scope('d_block_{}'.format(block_idx)):
                    if not reuse:
                        print('D block {} input shape: {}'
                              ''.format(block_idx, input_.get_shape()),
                              end=' *** ')
                    bias_init = None
                    if self.bias_D_conv:
                        if not reuse:
                            print('biasing D conv', end=' *** ')
                        bias_init = tf.constant_initializer(0.)
                    downconv_init = tf.truncated_normal_initializer(stddev=0.02)
                    hi_a = downconv(input_, nfmaps, kwidth=kwidth, pool=pooling,
                                    init=downconv_init, bias_init=bias_init)
                    if not reuse:
                        print('downconved shape: {} '
                              ''.format(hi_a.get_shape()), end=' *** ')
                    if bnorm:
                        if not reuse:
                            print('Applying VBN', end=' *** ')
                        hi_a = self.vbn(hi_a, '{}/d_vbn_{}'.format(name_disc, block_idx))
                    if activation == 'leakyrelu':
                        if not reuse:
                            print('Applying Lrelu', end=' *** ')
                        hi = leakyrelu(hi_a)
                    elif activation == 'relu':
                        if not reuse:
                            print('Applying Relu', end=' *** ')
                        hi = tf.nn.relu(hi_a)
                    else:
                        raise ValueError('Unrecognized activation {} '
                                         'in D'.format(activation))
                    return hi
            beg_size = self.canvas_size
            # apply input noisy layer to real and fake samples
            zstack = tf.cast(zstack, tf.float32)
            hi = gaussian_noise_layer(hi, self.disc_noise_std / 10.)
            if not reuse:
                print('*** Discriminator summary ***')
            for block_idx, fmap in enumerate(fmaps):
                dimension = hi.get_shape().as_list()[1]
                zconcat = zstack*tf.ones([self.batch_size, dimension, zdim])
                hi = tf.concat(values=[hi,zconcat],axis=2)
                if block_idx in large:
                    pooling = 4
                else :
                    pooling = 2
                hi = disc_block(block_idx, hi, 31, 
                                fmaps[block_idx],
                                True, 'leakyrelu', pooling)
                if not reuse:
                    print()
            if not reuse:
                print('discriminator deconved shape: ', hi.get_shape())
            hi_f = flatten(hi)
            #hi_f = tf.nn.dropout(hi_f, self.keep_prob_var)
            d_logit_out = conv1d(hi, kwidth=1, num_kernels=10,
                                 init=tf.truncated_normal_initializer(stddev=0.02),
                                 name='logits_conv')
            d_logit_out = tf.squeeze(d_logit_out)
            d_logit_out = fully_connected(d_logit_out, 1, activation_fn=None)
            if not reuse:
                print('discriminator output shape: ', d_logit_out.get_shape())
                print('*****************************')
            return d_logit_out
