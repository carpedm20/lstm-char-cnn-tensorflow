import tensorflow as tf

from .ops import conv2d
from base import Model

class TDNN(Model):
  """Time-delayed Nueral Network (cf. http://arxiv.org/abs/1508.06615v4)
  """
  def __init__(self, input_, embed_dim=650,
               feature_maps=[50, 100, 150, 200, 200, 200, 200],
               kernels=[1,2,3,4,5,6,7], checkpoint_dir="checkpoint",
               forward_only=False):
    """Initialize the parameters for TDNN

    Args:
      embed_dim: the dimensionality of the inputs
      feature_maps: list of feature maps (for each kernel width)
      kernels: list of # of kernels (width)
    """
    self.embed_dim = embed_dim
    self.feature_maps = feature_maps
    self.kernels = kernels

    # [batch_size x seq_length x embed_dim x 1]
    input_ = tf.expand_dims(input_, -1)

    layers = []
    for idx, kernel_dim in enumerate(kernels):
      reduced_length = input_.get_shape()[1] - kernel_dim + 1

      # [batch_size x seq_length x embed_dim x feature_map_dim]
      conv = conv2d(input_, feature_maps[idx], kernel_dim , self.embed_dim,
                    name="kernel%d" % idx)

      # [batch_size x 1 x 1 x feature_map_dim]
      pool = tf.nn.max_pool(tf.tanh(conv), [1, reduced_length, 1, 1], [1, 1, 1, 1], 'VALID')

      layers.append(tf.squeeze(pool))

    if len(kernels) > 1:
      self.output = tf.concat(1, layers)
    else:
      self.output = layers[0]
