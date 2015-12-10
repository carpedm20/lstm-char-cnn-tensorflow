import tensorflow as tf

from utils import *

class TDNN(object):
    def __init__(config):
        length = config.get('length', 100)
        input_size = config.get('input_size', 100)
        feature_maps = config.get('feature_maps', 100)
        kernels = config.get('kernels', 100)

    def build_model():
        x = tf.nn.placeholder(tf.int32, [None, length, input_size])
        x = tf.reshape(x, [-1, 1, length, input_size])

        layers = []
        for idx, kernel in enumerate(kernels):
            reduced_length = length - kernel + 1
            conv = tf.conv2d(x, weight([input_size, kernel, 1, feature_maps[idx]]), [1, 1, 1, 1], \
                            padding='SAME', name='conv_%d_%d' % (kernel, feature_maps[idx]))
            pool = tf.squeeze(tf.nn.max_pool(conv, [1, 1, reduced_length, 1], [1, 1, 1, 1], 'SAME'))
            layers.append(pool)

        if len(kernels) > 1:
            pass
        else:
            y = layers[0]

        return x, y

class LSTMTDNN(object):
    def __init__(config):
        pass
