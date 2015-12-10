import os

from models import *
from utils import *

config = {
  'cudnn' : 0,
  'max_epochs' : 25,
  'seed' : 3435,
  'batch_size' : 20,
  'highway_layers' : 2,
  'gpuid' : -1,
  'EOS' : "+",
  'char_vec_size' : 15,
  'learning_rate_decay' : 0.5,
  'num_layers' : 2,
  'max_grad_norm' : 5,
  'savefile' : "char",
  'kernels' : "{1,2,3,4,5,6,7}",
  'use_chars' : 1,
  'param_init' : 0.05,
  'checkpoint_dir' : "cv",
  'data_dir' : "data/ptb",
  'seq_length' : 35,
  'max_word_l' : 65,
  'feature_maps' : "{50,100,150,200,200,200,200}",
  'hsm' : 0,
  'word_vec_size' : 650,
  'rnn_size' : 650,
  'decay_when' : 1,
  'print_every' : 500,
  'use_words' : 0,
  'learning_rate' : 1,
  'batch_norm' : 0,
  'dropout' : 0.5,
  'time' : 0,
  'save_every' : 5,
}
