# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import pickle


import nltk
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Download the data.
#url = 'http://mattmahoney.net/dc/'


# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = nltk.word_tokenize( f.read(f.namelist()[0] ) )
    #data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

vocabulary = read_data('./All_hyphenated_clean.zip')
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
#vocabulary_size = 50000
vocabulary_size = len(np.unique(vocabulary))
print("vocabulary_size: ", vocabulary_size)

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  #count = [['UNK', -1]]
#  count = [[]]
#  count.extend(collections.Counter(words).most_common(n_words))
  dictionary = dict()
#  for word, _ in count:
#    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  count = 0

  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      #print ("unknown " + word)
      index = count
      dictionary[word] = index
      count += 1
    data.append(index)
    
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)

glove_300d_pickle = None
glove_path = "Hotel_data_average_embeddings.pickle"
with open(glove_path, 'rb') as handle:
    glove_300d_pickle = pickle.load(handle)


def dump_data(variable, path):
    with open(path, "wb") as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)


print ("Loaded pickle")

def get_word_embedding(word):
    if word in glove_300d_pickle:
        return glove_300d_pickle[word]
    else:
#        print(word)
        emb = ( ( 2 )*np.random.rand(1, 300) - 1 )
        return emb


#pretrained_embeddings = None
pretrained_embeddings = np.ndarray(shape=(len(reverse_dictionary), 300), dtype=np.float)

def get_pretrained_embeddings(reverse_dictionary, dictionary, dim_size):

    #pretrained_embeddings = np.ndarray(shape=(len(reverse_dictionary), dim_size), dtype=np.float)
    for i in range(len(reverse_dictionary)):
        word = reverse_dictionary[i]
        word_embedding = get_word_embedding(word)
        pretrained_embeddings[i] = np.asarray(word_embedding)




get_pretrained_embeddings(reverse_dictionary, dictionary, 300)

#print ("pretrained_embeddings " + pretrained_embeddings.shape)
#print ("vocabulary_shape " + vocab_size)

print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


dump_data(data, "data_pk.pickle")
print ("Saved data_list")

dump_data(vocabulary,"vocabulary_pk.pickle" )
print ("Saved vocabulary")

dump_data(reverse_dictionary,"reverse_pk.pickle" )
print ("Saved reverse dictionary")

np.save("pretrained_numpy", pretrained_embeddings)

dump_data(dictionary, "dict_pk.pickle")
print("Saved dictionary")

del vocabulary  # Hint to reduce memory.



