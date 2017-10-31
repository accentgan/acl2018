from __future__ import print_function
import tensorflow as tf
from ops import *
import numpy as np


def pre_emph(x, coeff=0.920):
    x0 = tf.reshape(x[0], [1,])
    diff = x[1:] - coeff * x[:-1]
    concat = tf.concat(axis=0, values=[x0, diff])
    return concat

def de_emph(y, coeff=0.920):
    if coeff <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coeff * x[n - 1] + y[n]
    return x

def read_and_decode(filename_queue, canvas_size, preemph=0., typeRun="vctk"):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'wav_raw': tf.FixedLenFeature([], tf.string),
                'noisy_raw': tf.FixedLenFeature([], tf.string),
                'class_val' : tf.FixedLenFeature([], tf.string),
                'class_dash' : tf.FixedLenFeature([], tf.string),
                'embedding_dash' : tf.FixedLenFeature([], tf.string)
            })
    wave = tf.decode_raw(features['wav_raw'], tf.int32)
    wave.set_shape(canvas_size)
    wave = (2./62020320.) * tf.cast((wave - 32767), tf.float32) + 1.
    noisy = tf.decode_raw(features['noisy_raw'], tf.int32)
    noisy.set_shape(canvas_size)
    noisy = (2./62020320.) * tf.cast((noisy - 32767), tf.float32) + 1.
    class_val = tf.decode_raw(features['class_val'], tf.int8)
    if (typeRun == "vctk"):	class_val.set_shape([20]) 
    class_dash = tf.decode_raw(features['class_dash'], tf.int8)
    if (typeRun == "vctk"):	class_val.set_shape([20]) 
    embedding_dash = tf.decode_raw(features['embedding_dash'],tf.float32)
    embedding_dash.set_shape([256])

    return wave, noisy, class_val, class_dash, embedding_dash
def read_and_decode_gpu(filename_queue, canvas_size, preemph=0., slice_num=4, typeRun="vctk"):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'wav_raw': tf.FixedLenFeature([], tf.string),
                'noisy_raw': tf.FixedLenFeature([], tf.string),
                'class_val' : tf.FixedLenFeature([], tf.string),
                'class_dash' : tf.FixedLenFeature([], tf.string),
                'embedding_dash' : tf.FixedLenFeature([], tf.string)
            })
    wave = tf.decode_raw(features['wav_raw'], tf.int32)
    wave.set_shape(slice_num*canvas_size)
    wave = (2./62020320.) * tf.cast((wave - 32767), tf.float32) + 1.
    wave = tf.reshape(wave, [slice_num, canvas_size])
    noisy = tf.decode_raw(features['noisy_raw'], tf.int32)
    noisy.set_shape(slice_num*canvas_size)
    noisy = (2./62020320.) * tf.cast((noisy - 32767), tf.float32) + 1.
    noisy = tf.reshape(noisy, shape=[slice_num, canvas_size])
    class_val = tf.decode_raw(features['class_val'], tf.int8)
    if (typeRun == "vctk"):	class_val.set_shape([20]) 
    class_dash = tf.decode_raw(features['class_dash'], tf.int8)
    if (typeRun == "vctk"):	class_dash.set_shape([20])
    embedding_dash = tf.decode_raw(features['embedding_dash'],tf.float32)
    embedding_dash.set_shape([256])

    return wave, noisy, class_val, class_dash, embedding_dash
