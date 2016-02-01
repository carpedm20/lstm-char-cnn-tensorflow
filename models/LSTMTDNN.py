import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

from ops import conv2d, batch_norm, highway
from base import Model
from TDNN import TDNN
from TDNN import TDNN
from batch_loader import BatchLoader

class LSTMTDNN(Model):
  """Time-delayed Nueral Network (cf. http://arxiv.org/abs/1508.06615v4)
  """
  def __init__(self,
               batch_size=20, rnn_size=100, layer_depth=2,
               word_vocab_size=None, char_vicab_size=None,
               word_embed_dim=100, char_embed_dim=15,
               feature_maps=[50, 100, 150, 200, 200, 200, 200],
               kernels=[1,2,3,4,5,6,7], seq_length=35,
               use_word=False, use_char=True, hsm=False,
               highway_layers=2, dropout_prob=0.5, use_batch_norm=True,
               checkpoint_dir="checkpoint", forward_only=False):
    """Initialize the parameters for LSTM TDNN

    Args:
      rnn_size: the dimensionality of hidden layers
      layer_depth: # of depth in LSTM layers
      batch_size: size of batch per epoch
      word_vocab_size: # of words in the vocabulary
      word_embed_dim: the dimensionality of word embeddings
      char_vocab_size: # of character in the vocabulary
      char_embed_dim: the dimensionality of character embeddings
      feature_maps: list of feature maps (for each kernel width)
      kernels: list of kernel widths
      seq_length: max length of a word
      use_word: whether to use word embeddings or not
      use_char: whether to use character embeddings or not
      highway_layers: # of highway layers to use
      dropout_prob: the probability of dropout
      use_batch_norm: whether to use batch normalization or not
      hsm: whether to use hierarchical softmax
    """
    self.batch_size = batch_size
    self.rnn_size = rnn_size
    self.layer_depth = layer_depth
    self.word_embed_dim = word_embed_dim
    self.word_vocab_size = word_vocab_size
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

  def prepare_model(self):
    with tf.variable_scope("LSTMTDNN"):
      self.char_indices = tf.placeholder(tf.int32, [self.batch_size, self.max_word_length])
      self.word_indices = tf.placeholder(tf.int32, [self.batch_size, 1])

      if self.use_char:
        char_W = tf.get_variable("char_embed",
            [self.char_vocab_size, self.char_embed_dim])
      else:
        word_W = tf.get_variable("word_embed",
            [self.word_vocab_size, self.word_embed_dim])

      if self.use_char:
        # [batch_size x word_max_length, char_embed]
        char_embed = tf.nn.embedding_lookup(char_W, self.char_indices)

        char_cnn = TDNN(char_embed, self.char_embed_dim, self.feature_maps, self.kernels)

        if self.use_word:
          word_embed = tf.embedding_lookup(word_W, self.word_indices)
          input_ = tf.concat(1, char_cnn.output, word_embed)
        else:
          input_ = char_cnn.output
      else:
        input_ = tf.embedding_lookup(word_W, self.word_indices)

      if self.use_batch_norm:
        bn = batch_norm()
        norm_output = bn(tf.expand_dims(tf.expand_dims(input_, 1), 1))
        input_ = tf.squeeze(norm_output)

      if highway:
        #input_ = highway(input_, input_dim_length, self.highway_layers, 0)
        input_ = highway(input_, input_.get_shape()[1], self.highway_layers, 0)

      self.cell = rnn_cell.BasicLSTMCell(self.rnn_size)
      self.stacked_cell = rnn_cell.MultiRNNCell([self.cell] * self.layer_depth)

      outputs, states = rnn.rnn(self.cell,
                                input_,
                                seq_length=self.seq_length,
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

  def train(self, sess, dataset, max_word_length=65, 
            epoch=25, batch_size=20, learning_rate=1, 
            decay=0.5, data_dir='data'):
    loader = BatchLoader(data_dir, dataset, batch_size, self.seq_length, max_word_length)
    print('Word vocab size: %d, Char vocab size: %d, Max word length (incl. padding): %d' % \
        (len(loader.idx2word), len(loader.idx2char), loader.max_word_length))

    self.max_word_length = loader.max_word_length
    self.char_vocab_size = len(loader.idx2char)
    self.word_vocab_size = len(loader.idx2word)

    self.prepare_model()
