import os
from glob import glob
import tensorflow as tf

class Model(object):
  """Abstract object representing an Reader model."""
  def __init__(self):
    self.vocab = None
    self.data = None

  def train(self, sess, vocab_size, epoch=25,
            learning_rate=1, decay=0.5,
            data_dir='data', dataset_name='ptb'):
    pass
    grads_and_vars = self.opt.compute_gradients(self.loss,params)
    clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                               for gv in grads_and_vars]

  def save(self, checkpoint_dir, dataset_name):
    self.saver = tf.train.Saver()

    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__ or "Reader"
    if self.batch_size:
      model_dir = "%s_%s" % (dataset_name, self.batch_size)
    else:
      model_dir = dataset_name

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name))

  def load(self, checkpoint_dir, dataset_name):
    self.saver = tf.train.Saver()

    print(" [*] Loading checkpoints...")
    if self.batch_size:
      model_dir = "%s_%s" % (dataset_name, self.batch_size)
    else:
      model_dir = dataset_name
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      return True
    else:
      return False
