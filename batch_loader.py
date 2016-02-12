import re
import os
import math
import pickle
import numpy as np


def save(fname, obj):
  with open(fname, 'w') as f:
    pickle.dump(obj, f)


def load(fname):
  with open(fname, 'r') as f:
    return pickle.load(f)


class BatchLoader(object):
  
  def __init__(self, data_dir, dataset_name, batch_size, seq_length, max_word_length):
    train_fname = os.path.join(data_dir, dataset_name, 'train.txt')
    valid_fname = os.path.join(data_dir, dataset_name, 'valid.txt')
    test_fname = os.path.join(data_dir, dataset_name, 'test.txt')
    input_fnames = [train_fname, valid_fname, test_fname]

    vocab_fname = os.path.join(data_dir, dataset_name, 'vocab.pkl')
    tensor_fname = os.path.join(data_dir, dataset_name, 'data.pkl')
    char_fname = os.path.join(data_dir, dataset_name, 'data_char.pkl')

    if not os.path.exists(vocab_fname) or not os.path.exists(tensor_fname) or not os.path.exists(char_fname):
      print("Creating vocab...")
      self.text_to_tensor(input_fnames, vocab_fname, tensor_fname, char_fname, max_word_length)

    print("Loading vocab...")
    all_data = load(tensor_fname)
    all_data_char = load(char_fname)
    self.idx2word, self.word2idx, self.idx2char, self.char2idx = load(vocab_fname)
    vocab_size = len(self.idx2word)

    print("Word vocab size: %d, Char vocab size: %d" % (len(self.idx2word), len(self.idx2char)))
    self.max_word_length = all_data_char[0].shape[1]
    self.sizes = []
    self.all_batches = []

    print("Reshaping tensors...")
    for split, data in enumerate(all_data): # split = 0:train, 1:valid, 2:test
      length = data.shape[0]
      #if length % (batch_size * seq_length) != 0 and split < 2:
      data = data[: batch_size * seq_length * math.floor(length / (batch_size * seq_length))]
      ydata = np.zeros_like(data)
      ydata[:-1] = data[1:].copy()
      ydata[-1] = data[0].copy()
      data_char = np.zeros([data.shape[0], self.max_word_length])

      for idx in xrange(data.shape[0]):
        data_char[idx] = all_data_char[split][idx]

      if split < 2:
        x_batches = list(data.reshape([-1, batch_size, seq_length]))
        y_batches = list(ydata.reshape([-1, batch_size, seq_length]))
        x_char_batches = list(data_char.reshape([-1, batch_size, seq_length, self.max_word_length]))
        self.sizes.append(len(x_batches))
      else:
        x_batches = list(data.reshape([-1, batch_size, seq_length]))
        y_batches = list(ydata.reshape([-1, batch_size, seq_length]))
        x_char_batches = list(data_char.reshape([-1, batch_size, seq_length, self.max_word_length]))
        self.sizes.append(len(x_batches))
        # x_batches = np.tile(data, (batch_size, 1))
        # y_batches = np.tile(ydata, (batch_size, 1))
        # x_char_batches = np.tile(data_char, (batch_size, 1)).reshape(batch_size, -1, data_char.shape[1])
        # self.sizes.append(1)
      self.all_batches.append([x_batches, y_batches, x_char_batches])

    self.batch_idx = [0, 0, 0]
    print("data load done. Number of batches in train: %d, val: %d, test: %d" \
        % (self.sizes[0], self.sizes[1], self.sizes[2]))

  def next_batch(self, split_idx):
    # cycle around to beginning
    if self.batch_idx[split_idx] >= self.sizes[split_idx]:
      self.batch_idx[split_idx] = 0
    idx = self.batch_idx[split_idx]
    self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
    return self.all_batches[split_idx][0][idx], \
           self.all_batches[split_idx][1][idx], \
           self.all_batches[split_idx][2][idx]

  def reset_batch_pointer(self, split_idx, batch_idx=None):
    if batch_idx == None:
      batch_idx = 0
    self.batch_idx[split_idx] = batch_idx

  def text_to_tensor(self, input_files, vocab_fname, tensor_fname, char_fname, max_word_length):
    max_word_length_tmp = 0
    counts = []

    for input_file in input_files:
      count = 0

      with open(input_file) as f:
        for line in f:
          line = line.replace('<unk>', '|')
          line = line.replace('}', '')
          line = line.replace('{', '')
          for word in line.split():
            max_word_length_tmp = max(max_word_length_tmp, len(word) + 2)
            count += 1

          count += 1 # for \n
      counts.append(count)

    print("After first pass of data, max word length is: %d" % max_word_length_tmp)
    print("Token count: train %d, val %d, test %d" % (counts[0], counts[1], counts[2]))

    max_word_length = min(max_word_length_tmp, max_word_length)

    char2idx = {' ':0, '{': 1, '}': 2}
    word2idx = {'<unk>': 0}
    idx2char = [' ', '{', '}']
    idx2word = ['<unk>']

    output_tensors = []
    output_chars = []

    for idx, input_file in enumerate(input_files):
      count = 0

      with open(input_file) as f:
        output_tensor = np.ndarray(counts[idx])
        output_char = np.ones([counts[idx], max_word_length])

        word_num = 0
        for line in f:
          line = line.replace('<unk>', '|')
          line = line.replace('}', '')
          line = line.replace('{', '')

          for word in line.split() + ['+']:
            chars = [char2idx['{']]
            if word[0] == '|' and len(word) > 1:
              word = word[2:]
              output_tensor[word_num] = word2idx['|']
            else:
              if not word2idx.has_key(word):
                idx2word.append(word)
                word2idx[word] = len(idx2word) - 1
              output_tensor[word_num] = word2idx[word]

            for char in word:
              if not char2idx.has_key(char):
                idx2char.append(char)
                char2idx[char] = len(idx2char) - 1
              chars.append(char2idx[char])
            chars.append(char2idx['}'])

            if len(chars) == max_word_length:
              chars[-1] = char2idx['}']

            for idx in xrange(min(len(chars), max_word_length)):
              output_char[word_num][idx] = chars[idx]
            word_num += 1

        output_tensors.append(output_tensor)
        output_chars.append(output_char)

    save(vocab_fname, [idx2word, word2idx, idx2char, char2idx])
    save(tensor_fname, output_tensors)
    save(char_fname, output_chars)
