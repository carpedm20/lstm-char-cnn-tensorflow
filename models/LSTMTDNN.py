import tensorflow as tf

from ops import conv2d
from utils import *

class LSTMTDNN(object):
  """Time-delayed Nueral Network (cf. http://arxiv.org/abs/1508.06615v4)
  """
  def __init__(self, rnn_size, layer_size, vocab_size, vocab_embed_size,
               feature_maps, kernels, length, word_or_char, highway_layers,
               dropout_prob):
    """Initialize the parameters for LSTM TDNN

    Args:
      rnn_size: the dimensionality of hidden layers
      layer_size: # of layers
      vocab_size: # of words in the vocabulary
      vocab_embed_size: the dimensionality of word embeddings
      feature_maps: list of feature maps (for each kernel width)
      kernels: list of kernel widths
      length: max length of a word
      word_or_char: whether to use word or character embeddings
      highway_layers: # of highway layers to use
      dropout_prob: the probability of dropout
    """
    self.length = length
    self.input_size = input_size
    self.feature_maps = feature_maps
    self.kernels = kernels

    self.input_ = tf.placeholder(tf.float32, [None, self.length, self.input_size])
    embed = tf.get_variable("%s_embed" % word_or_char, [vocab_size, vocab_embed_size])

    if use_chars:
      char_embed = tf.embedding_lookup(embed, self.input_)

      char_cnn = TDNN(self.length, self.vocab_embed_size, self.feature_maps, self.kernels)
      input_size_length = ???

      if use_words:
        input_ = tf.concat(1, char_cnn.output, word_embed)
        input_size_length = input_size_length + word_embed_size
      else:
        input_ = char_cnn.output
    else:
      input_ = tf.embedding_lookup(embed, self.input_)
      input_size_length = vocab_embed_size

    for _ in xrange(self.layer_size):
      self.inputs.append(


    # [batch_size x length x input_size]
    input_ = tf.reshape(x, [-1, 1, self.length, self.input_size])

    layers = []
    for idx, kernel in enumerate(kernels):
      reduced_length = length - kernel + 1

      conv = conv2d(input_, feature_maps[idx], self.input_size, kernel, 1, 1, 0)
      pool = tf.squeeze(tf.nn.max_pool(conv, [1, 1, reduced_length, 1], [1, 1, 1, 1], 'SAME'))

      layers.append(pool)

    if len(kernels) > 1:
      self.output = tf.concat(1, layers)
    else:
      self.output = layers[0]
