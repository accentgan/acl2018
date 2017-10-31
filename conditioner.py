from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from ops import *
import numpy as np
from bnorm import VBN

class PhoneGenerator(object):
	def __init__(self, segan):
		self.segan = segan
	def __call__(self, noisy_w, is_ref, spk=None, z_on=False, do_prelu=False):
		segan = self.segan
		if hasattr(segan, "generator_built"):
			tf,get_veriable_scope().reuse_variables()
			make_vars=False
		else :
			make_vars=True
		print('*** Building Generator ***')
		in_dims = noisy_w.get_shape().as_list()
		h_i = noisy_w
		if len(in_dims) == 2:
			h_i = tf.expand_dims(noisy_w,-1)
		elif len(in_dims) < 2 or len(in_dims) > 3:
			raise ValueError("Generator input shape be 2-D or 3-D")
		kwidth = 31
		enc_layers = 7
		skips = []
		if is_ref and do_prelu : 
			alphas = []
		with tf.variable_scope("g_phone_ae"):
			for layer_idx, layer_depth in  enumerate(segan.pg_enc_depths):
				bias_init = None
				if segan.bias_downconv : 
					if is_ref : 
						print("Biasing downconv in G")
					bias_init = tf.constant_initializer(0.)
				h_i_down = downconv(h_i, layer_depth, kwidth=kwidth,
					init=tf.truncated_normal_initailizer(stddev=0.02),
					bias_init=bias_init,
					name="enc_{}".format(layer_idx))
				h_i = h_i_down
				if layer_idx < len(segan.g_enc_depths) - 1:
					skips.append(h_i)
				if do_prelu : 
					h_i = prelu(h_i, ref=is_ref, name="enc_prelu_{}".format(layer_idx))
					if is_ref :
						alpha_i = h_i[1]
						h_i = h_i[0]
						alphas.append(alpha_i)
				else :
					h_i = leakyrelu(h_i)
			zmid = h_i
		return tf.squeeze(zmid)
	def vbn(self, tensor, name):
		if not hasattr(self, name):
			vbn = VBN(tensor, name)
			setattr(self, name, vbn)
			return vbn.reference_output
		vbn = getattr(self, name)
		return vbn(tensor)
	def discriminator(self, phone, reuse=False):
		in_dims = phone.get_shape().as_list()
		hi = phone
		if len(in_dims) == 2:
			hi = tf.expand_dims(phone, -1)
		elif len(in_dims) < 2 or len(in_dims) > 3:
			raise ValueError('discriminator input must be 2-D or 3-D')
		batch_size = int(phone.get_shape()[0])
		with tf.variable_scope("d_phone") as scope:
			if reuse : 
				scope.reuse_variables()
			def disc_block(block_idx, input_, kwidth, nfmaps, bnorm, activation,  
				pooling=2):
				with tf.variable_scope("d_block_{}".format(block_idx)):
					bias_init=None
					if self.bias_D_conv:
						bias_init = tf.constant_initializer(0.)
					downconv_init = tf.truncated_normal_initailizer(stddev=0.02)
					hi_a = downconv(input_, nfmaps, kwidth=kwidth, pool=pooling,
						init=downconv_init, bias_init=bias_inits)
					if bnorm : 
						hi_a = self.vbn(hi_a, 'd_vbn_{}'.format(block_idx))
					if activation == 'leakyrelu' :
						hi = leakyrelu(hi_a)
					elif activation == 'relu' :
						hi = tf.nn.relu(hi_a)
					else :
						raise ValueError("Fucked up")
					return hi
			beg_size = phone.get_shape().as_list()[-1]
			hi = gaussian_noise_layer(hi, 0.02)
			for block_idx, fmaps in enumerate(self.d_phone_fmaps):
				dimension = hi.get_shape().as_list()[1]
				hi = disc_block(block_idx, hi, 31, self.d_num_fmaps[block_idx],
					True, 'leakyrelu')
			hi_f = flatten(hi)
			d_logit_out = conv1d(hi, kwidth=1, num_kernels=10, 
				init=tf.truncated_normal_initailizer(stddev=0.02),
				name="logits_conv")
			d_logit_out = tf.squeeze(d_logit_out)
			d_logit_out = tf.layers.dense(d_logit_out, 1, activation_fn=None)
			return d_logit_out

def ClassDiscriminator(tensor, reuse=True,layers=3):
	with tf.variable_scope("c_d_latent_class") as scope:
		if reuse :
			scope.reuse_variables()
		hi = tensor
		print(reuse)
		print(scope.reuse)
		for i in range(layers):
			hi = tf.layers.dense(inputs=hi,units=750, activation=tf.nn.relu, 
				name="layers_%d"%(i), reuse=scope.reuse)
			mean, vari = tf.nn.moments(hi, [0],keep_dims=True)
			hi =  tf.nn.batch_normalization(hi, mean, vari, offset=None, 
            	scale=None, variance_epsilon=1e-6)
		logit = tf.layers.dense(inputs=hi, units=1, activation=None, 
			name="logit_layer", reuse=scope.reuse)
		return logit

def EmbeddingDiscriminator(tensor, name=None, reuse=True, layers=5):
	with tf.variable_scope("c_d_latent_embedding") as scope:
		if reuse : 
			scope.reuse_variables()
		hi = tensor 
		for i in range(layers):
			hi = tf.layers.dense(inputs=hi, units=750, activation=tf.nn.relu,
				name="layers_%d"%(i), reuse=scope.reuse)
			mean, vari = tf.nn.moments(hi, [0], keep_dims=True)
			hi = tf.nn.batch_normalization(hi, mean, vari, offset=None, 
				scale=None, variance_epsilon=1e-6)
		logit = tf.layers.dense(inputs=hi, units=1, activation=None,
			name="logit_layer", reuse=scope.reuse)
		return logit

# class TextGenerator(object):
# 	def __init__(self, segan):
# 		self.segan = segan
# 	def __call__(self, noisy_w, is_ref, spk=None, z_on=False, do_prelu=False):
# 		segan = self.segan
# 		if hasattr(segan, "generator_built"):
# 			tf,get_veriable_scope().reuse_variables()
# 			make_vars=False
# 		else :
# 			make_vars=True
# 		print('*** Building Generator ***')
# 		in_dims = noisy_w.get_shape().as_list()
# 		h_i = noisy_w
# 		kwidth = 31
# 		enc_layers = 7
# 		skips = []
# 		if is_ref and do_prelu : 
# 			alphas = []
# 		with tf.variable_scope("g_phone_ae"):
# 			for layer_idx, layer_depth in  enumerate(segan.pg_enc_depths):
# 				bias_init = None
# 				h_i_down = downconv(h_i, layer_depth, kwidth=kwidth,
# 					init=tf.truncated_normal_initailizer(stddev=0.02),
# 					bias_init=bias_init,
# 					name="enc_{}".format(layer_idx))
# 				h_i = h_i_down
# 				if layer_idx < len(segan.g_enc_depths) - 1:
# 					skips.append(h_i)
# 				if do_prelu : 
# 					h_i = prelu(h_i, ref=is_ref, name="enc_prelu_{}".format(layer_idx))
# 					if is_ref :
# 						alpha_i = h_i[1]
# 						h_i = h_i[0]
# 						alphas.append(alpha_i)
# 				else :
# 					h_i = leakyrelu(h_i)
# 			zmid = h_i
# 		return tf.squeeze(zmid)
# 	def vbn(self, tensor, name):
# 		if not hasattr(self, name):
# 			vbn = VBN(tensor, name)
# 			setattr(self, name, vbn)
# 			return vbn.reference_output
# 		vbn = getattr(self, name)
# 		return vbn(tensor)
# 	def discriminator(self, phone, reuse=False):
# 		in_dims = phone.get_shape().as_list()
# 		hi = phone
# 		if len(in_dims) == 2:
# 			hi = tf.expand_dims(phone, -1)
# 		elif len(in_dims) < 2 or len(in_dims) > 3:
# 			raise ValueError('discriminator input must be 2-D or 3-D')
# 		batch_size = int(phone.get_shape()[0])
# 		with tf.variable_scope("d_phone") as scope:
# 			if reuse : 
# 				scope.reuse_variables()
# 			def disc_block(block_idx, input_, kwidth, nfmaps, bnorm, activation,  
# 				pooling=2):
# 				with tf.variable_scope("d_block_{}".format(block_idx)):
# 					bias_init=None
# 					if self.bias_D_conv:
# 						bias_init = tf.constant_initializer(0.)
# 					downconv_init = tf.truncated_normal_initailizer(stddev=0.02)
# 					hi_a = downconv(input_, nfmaps, kwidth=kwidth, pool=pooling,
# 						init=downconv_init, bias_init=bias_inits)
# 					if bnorm : 
# 						hi_a = self.vbn(hi_a, 'd_vbn_{}'.format(block_idx))
# 					if activation == 'leakyrelu' :
# 						hi = leakyrelu(hi_a)
# 					elif activation == 'relu' :
# 						hi = tf.nn.relu(hi_a)
# 					else :
# 						raise ValueError("Fucked up")
# 					return hi
# 			beg_size = phone.get_shape().as_list()[-1]
# 			hi = gaussian_noise_layer(hi, 0.02)
# 			for block_idx, fmaps in enumerate(self.d_phone_fmaps):
# 				dimension = hi.get_shape().as_list()[1]
# 				hi = disc_block(block_idx, hi, 31, self.d_num_fmaps[block_idx],
# 					True, 'leakyrelu')
# 			hi_f = flatten(hi)
# 			d_logit_out = conv1d(hi, kwidth=1, num_kernels=10, 
# 				init=tf.truncated_normal_initailizer(stddev=0.02),
# 				name="logits_conv")
# 			d_logit_out = tf.squeeze(d_logit_out)
# 			d_logit_out = dense(d_logit_out, 1, activation_fn=None)
# 			return d_logit_out
