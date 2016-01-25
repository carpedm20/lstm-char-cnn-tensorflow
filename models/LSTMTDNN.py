import tensorflow as tf

from ops import conv2d
from utils import *

class LSTMTDNN(object):
  """Time-delayed Nueral Network (cf. http://arxiv.org/abs/1508.06615v4)
  """
  def __init__(self, rnn_size, layer_size, vocab_size, vocab_embed_size,
               feature_maps, kernels, length, word_or_char, highway_layers,
               dropout_prob, use_batchnorm, hsm):
    """Initialize the parameters for LSTM TDNN

    Args:
      rnn_size: the dimensionality of hidden layers
      layer_size: # of layers
      vocab_size: # of words in the vocabulary
      vocab_embed_size: the dimensionality of word embeddings
      feature_maps: list of feature maps (for each kernel width)
      kernels: list of kernel widths
      length: max length of a word
      use_word: whether to use word embeddings or not
      use_char: whether to use character embeddings or not
      highway_layers: # of highway layers to use
      dropout_prob: the probability of dropout
      use_batchnorm: whether to use batch normalization or not
    """
    self.length = length
    self.input_size = input_size
    self.feature_maps = feature_maps
    self.kernels = kernels

    self.outputs = []
    states = []

    with tf.variable_scope("LSTMTDNN"):
      self.input_ = tf.placeholder(tf.float32, [None, self.length, self.input_size])
      embed = tf.get_variable("%s_embed" % word_or_char, [vocab_size, vocab_embed_size])

      if use_chars:
        char_embed = tf.embedding_lookup(embed, self.input_)

        char_cnn = TDNN(self.length, self.vocab_embed_size, self.feature_maps, self.kernels)
        input_size_length = sum(feature_maps)

        if use_words:
          input_ = tf.concat(1, char_cnn.output, word_embed)
          input_size_length = input_size_length + word_embed_size
        else:
          input_ = char_cnn.output
      else:
        input_ = tf.embedding_lookup(embed, self.input_)
        input_size_length = vocab_embed_size

      if use_batch_norm:
        bn = batch_norm(batch_size, name='bn')
        self.batch_norms.append(bn)

        input_ = bn(input_)

      self.cell = BasicLSTMCell(self.rnn_size)
      self.stacked_cell = rnn_cell.MultiRNNCell([self.cell] * self.layer_size)

      outputs, states = rnn.rnn(self.cell,
                                input_,
                                dtype=tf.float32)

      top_h = outputs[-1]
      if dropout_prob > 0:
        top_h = tf.nn.dropout(top_h, dropout_prob)

      if hsm > 0:
        self.output = top_h
      else:
        proj = rnn_cell.linear(top_h, vocab_size)
        log_softmax = tf.log(tf.nn.softmax(proj))
        self.output = log_softmax
