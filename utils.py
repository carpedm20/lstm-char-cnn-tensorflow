import re
import os
import pprint
import pickle
import numpy as np

import gensim

pp = pprint.PrettyPrinter()

def save(fname, obj):
  with open(fname, 'w') as f:
    pickle.save(fname, obj)

def load(fname):
  with open(fname, 'r') as f:
    return pickle.load(fname)

def batch_loader(data_dir, dataset_name, batch_size, seq_length, max_word_length):
  train_file = os.path.join(data_dir, dataset_name, 'train.txt')
  valid_file = os.path.join(data_dir, dataset_name, 'valid.txt')
  test_file = os.path.join(data_dir, dataset_name, 'test.txt')
  input_files = [train_file, valid_file, test_file]

  vocab_file = os.path.join(data_dir, dataset_name, 'vocab.t7')
  tensor_file = os.path.join(data_dir, dataset_name, 'data.t7')
  char_file = os.path.join(data_dir, dataset_name, 'data_char.t7')

  if not os.path.exists(vocab_file) or not os.path.exists(tensor_file) or not os.path.exists(char_file):
    text_to_tensor(input_files, vocab_file, tensor_file, char_file, max_word_length)

def text_to_tensor(input_files, vocab_file, tensor_file, char_file, max_word_length):
  max_word_length_tmp = 0
  counts = []

  for input_file in input_files:
    count = 0

    with open(input_file) as f:
      for line in f:
        line = line.replace('<unk>', '|')
        line = line.replace('}', ' ')
        line = line.replace('{', ' ')
        for word in line.split():
          max_word_length_tmp = max(max_word_length_tmp, len(word) + 2)
          count += 1

        count += 1 # for \n
    counts.append(count)

  print(" [*] max word length : %d" % max_word_length_tmp)
  print(" [*] # of token, train: %s, val: %s, test: %s" % (counts[0], counts[1], counts[2]))

  max_word_length = min(max_word_length_tmp, max_word_length)

  char2idx = {' ':0, '{': 1, '}': 2}
  word2idx = {'<unk>': 0}
  idx2char = []
  idx2word = []

  output_tensors = []
  output_chars = []

  for idx, input_file in enumerate(input_files):
    count = 0

    with open(input_file) as f:
      output_tensor = np.ndarray(counts[idx])
      output_char = np.ones(counts[idx], max_word_length, dtype=np.float32)

      word_num = 0
      for line in f:
        line = line.replace('<unk>', '|')
        line = line.replace('}', ' ')
        line = line.replace('{', ' ')
        for word in line.split():
          word_num += 1
          chars = [char2idx['{']]
          if word[0] == '|' and len(word) > 1:
            word = word[3:]
            output_tensor[word_num] = word2idx['|']
          else:
            if not word2idx.has_key(word):
              idx2word.append(word)
              word2idx[word] = len(idx2word) - 1
            output_tensor[word_num] = word2idx[word]

          for c in word:
            if not char2idx.has_key(char):
              idx2char.append(char)
              char2idx[char] = len(idx2char) - 1
            char.append(char2idx[char])

            for idx in xrange(min(len(char), max_word_length)):
              output_char[word_num][idx] = char[idx]

            if len(char) == max_word_length:
              char.append(char2idx['}'])

          output_tensors.append(output_tensor)
          output_chars.append(output_char)

  save(vocab_file, [idx2word, word2idx, idx2char, char2idx])
  save(tensor_file, output_tensors)
  save(char_file, output_chars)

if __name__ == '__main__':
  batch_loader('data', 'ptb', 20, 40, 20)
