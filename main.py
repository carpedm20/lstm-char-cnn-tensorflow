import os
import numpy as np
import tensorflow as tf

from models import LSTMTDNN
from utils import *

from utils import pp

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_integer("word_embed_dim", 650, "The dimension of word embedding matrix [650]")
flags.DEFINE_integer("char_embed_dim", 15, "The dimension of char embedding matrix [15]")
flags.DEFINE_integer("max_word_length", 65, "The maximum length of word [65]")
flags.DEFINE_integer("batch_size", 100, "The size of batch images [100]")
flags.DEFINE_integer("seq_length", 35, "The # of timesteps to unroll for [35]")
flags.DEFINE_float("learning_rate", 1.0, "Learning rate [1.0]")
flags.DEFINE_float("decay", 0.5, "Decay of SGD [0.5]")
flags.DEFINE_float("dropout_prob", 0.5, "Probability of dropout layer [0.5]")
flags.DEFINE_string("feature_maps", "[50,100,150,200,200,200,200]", "The # of feature maps in CNN [50,100,150,200,200,200,200]")
flags.DEFINE_string("kernels", "[1,2,3,4,5,6,7]", "The width of CNN kernels [1,2,3,4,5,6,7]")
flags.DEFINE_string("model", "LSTMTDNN", "The type of model to train and test [LSTM, LSTMTDNN]")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("dataset", "ptb", "The name of dataset [ptb]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("forward_only", False, "True for forward only, False for training [False]")
flags.DEFINE_boolean("use_char", True, "Use character-level language model [True]")
flags.DEFINE_boolean("use_word", False, "Use word-level language [False]")
FLAGS = flags.FLAGS

model_dict = {
  'LSTM': None,
  'LSTMTDNN': LSTMTDNN,
}

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    print(" [*] Creating checkpoint directory...")
    os.makedirs(FLAGS.checkpoint_dir)

  with tf.Session() as sess:
    model = model_dict[FLAGS.model](sess, checkpoint_dir=FLAGS.checkpoint_dir,
                                    seq_length=FLAGS.seq_length,
                                    word_embed_dim=FLAGS.word_embed_dim,
                                    char_embed_dim=FLAGS.char_embed_dim,
                                    feature_maps=eval(FLAGS.feature_maps),
                                    kernels=eval(FLAGS.kernels),
                                    batch_size=FLAGS.batch_size,
                                    dropout_prob=FLAGS.dropout_prob,
                                    max_word_length=FLAGS.max_word_length,
                                    forward_only=FLAGS.forward_only,
                                    dataset_name=FLAGS.dataset,
                                    use_char=FLAGS.use_char,
                                    use_word=FLAGS.use_word,
                                    data_dir=FLAGS.data_dir)

    if not FLAGS.forward_only:
      model.run(FLAGS.epoch, FLAGS.learning_rate, FLAGS.decay)
    else:
      test_loss = model.test(2)
      print(" [*] Test loss: %2.6f, perplexity: %2.6f" % (test_loss, np.exp(test_loss)))

if __name__ == '__main__':
  tf.app.run()
