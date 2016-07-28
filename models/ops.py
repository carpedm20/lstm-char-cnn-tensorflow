import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

def conv2d(input_, output_dim, k_h, k_w,
           stddev=0.02, name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID')
    return conv

def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
  """Highway Network (cf. http://arxiv.org/abs/1505.00387).
  
  t = sigmoid(Wy + b)
  z = t * g(Wy + b) + (1 - t) * y

  where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
  """
  output = input_
  for idx in xrange(layer_size):
    output = f(tf.nn.rnn_cell._linear(output, size, 0, scope='output_lin_%d' % idx))

    transform_gate = tf.sigmoid(
        tf.nn.rnn_cell._linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)
    carry_gate = 1. - transform_gate

    output = transform_gate * output + carry_gate * input_

  return output

class batch_norm(object):
  """Code modification of http://stackoverflow.com/a/33950177"""
  def __init__(self, epsilon=1e-5, momentum = 0.1, name="batch_norm"):
    with tf.variable_scope(name) as scope:
      self.epsilon = epsilon
      self.momentum = momentum

      self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
      self.name=name

  def __call__(self, x, train=True):
    shape = x.get_shape().as_list()

    with tf.variable_scope(self.name) as scope:
      self.gamma = tf.get_variable("gamma", [shape[-1]],
          initializer=tf.random_normal_initializer(1., 0.02))
      self.beta = tf.get_variable("beta", [shape[-1]],
          initializer=tf.constant_initializer(0.))

      mean, variance = tf.nn.moments(x, [0, 1, 2])

      return tf.nn.batch_norm_with_global_normalization(
        x, mean, variance, self.beta, self.gamma, self.epsilon,
        scale_after_normalization=True)
