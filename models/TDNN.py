import tensorflow as tf

from .ops import conv2d
from base import Model

class TDNN(Model):
  """Time-delayed Nueral Network (cf. http://arxiv.org/abs/1508.06615v4)
  """
  def __init__(self, length=65, input_size=650,
               feature_maps=[50, 100, 150, 200, 200, 200, 200],
               kernels=[1,2,3,4,5,6,7], checkpoint_dir="checkpoint",
               forward_only=False):
    """Initialize the parameters for TDNN

    Args:
      length: length of sentences/words (zero padded to be of same length)
      input_size: the dimensionality of the inputs
      feature_maps: list of feature maps (for each kernel width)
      kernels: list of kernel widths
    """
    self.length = length
    self.input_size = input_size
    self.feature_maps = feature_maps
    self.kernels = kernels

    # [batch_size x length x input_size]
    self.input_ = tf.placeholder(tf.float32, [None, self.length, self.input_size])
    input_ = tf.reshape(self.input_, [-1, self.input_size, self.length, 1])

    layers = []
    for idx, kernel in enumerate(kernels):
      reduced_length = length - kernel + 1

      conv = conv2d(input_, feature_maps[idx], self.input_size,
                    kernel, 1, 1, 0, name="kernel%d" % idx)
      pool = tf.squeeze(tf.nn.max_pool(conv, [1, 1, reduced_length, 1], [1, 1, 1, 1], 'SAME'))

      layers.append(pool)

    if len(kernels) > 1:
      self.output = tf.concat(1, layers)
    else:
      self.output = layers[0]
