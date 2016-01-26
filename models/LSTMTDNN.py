import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

from ops import conv2d, batch_norm
from base import Model
from TDNN import TDNN

class LSTMTDNN(Model):
  """Time-delayed Nueral Network (cf. http://arxiv.org/abs/1508.06615v4)
  """
  def __init__(self, vocab_size, batch_size=10, rnn_size=100, layer_depth=2,
               word_embed_dim=100, char_embed_dim=15,
               feature_maps=[50, 100, 150, 200, 200, 200, 200],
               kernels=[1,2,3,4,5,6,7], seq_length=65,
               use_word=False, use_char=True,
               highway_layers=1, dropout_prob=0.5, use_batch_norm=True,
               checkpoint_dir="checkpoint", forward_only=False):
    """Initialize the parameters for LSTM TDNN

    Args:
      rnn_size: the dimensionality of hidden layers
      layer_depth: # of depth in LSTM layers
      vocab_size: # of words in the vocabulary
      batch_size: size of batch per epoch
      word_embed_dim: the dimensionality of word embeddings
      char_embed_dim: the dimensionality of character embeddings
      feature_maps: list of feature maps (for each kernel width)
      kernels: list of kernel widths
      seq_length: max length of a word
      use_word: whether to use word embeddings or not
      use_char: whether to use character embeddings or not
      highway_layers: # of highway layers to use
      dropout_prob: the probability of dropout
      use_batch_norm: whether to use batch normalization or not
    """
    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.rnn_size = rnn_size
    self.layer_depth = layer_depth
    self.word_embed_dim = word_embed_dim
    self.char_embed_dim = char_embed_dim
    self.feature_maps = feature_maps
    self.kernels = kernels
    self.seq_length = seq_length
    self.use_char = use_char
    self.use_word = use_word
    self.highway_layers = highway_layers
    self.dropout_prob = dropout_prob
    self.use_batch_norm = use_batch_norm
    self.kernels = kernels

    self.outputs = []
    states = []

    with tf.variable_scope("LSTMTDNN"):
      self.input_ = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
      if use_char:
        embed = tf.get_variable("char_embed", [vocab_size, char_embed_dim])
      else:
        embed = tf.get_variable("word_embed", [vocab_size, word_embed_dim])

      if use_char:
        char_embed = tf.nn.embedding_lookup(embed, self.input_)

        char_cnn = TDNN(char_embed, self.seq_length, 
            self.char_embed_dim, self.feature_maps, self.kernels)
        input_dim_length = sum(feature_maps)

        if use_word:
          input_ = tf.concat(1, char_cnn.output, word_embed)
          input_dim_length = input_dim_length + word_embed_dim
        else:
          input_ = char_cnn.output
      else:
        input_ = tf.embedding_lookup(embed, self.input_)
        input_dim_length = word_embed_dim

      if use_batch_norm:
        bn = batch_norm()
        norm_output = bn(tf.expand_dims(tf.expand_dims(input_, 1), 1))
        input_ = tf.squeeze(norm_output)

      self.cell = rnn_cell.BasicLSTMCell(self.rnn_size)
      self.stacked_cell = rnn_cell.MultiRNNCell([self.cell] * self.layer_depth)

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
