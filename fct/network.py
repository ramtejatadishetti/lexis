import tensorflow as tf
import numpy as np
import pickle

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

feature_size = 10
vocabulary_size = 10
embedding_size = 300
batch_size = 1
num_sampled = 64    # Number of negative examples to sample.

global_dict_for_indices = {}
count_of_tokens ={}


phrase_dict = {}
INDEX_MIN = 0
INDEX_MAX = 24
CONTEXT_MIN = 25
CONTEXT_MAX = 28

W1_INDEX = 0
W2_INDEX = 25
C1_INDEX = 50
C2_INDEX = 51
C3_INDEX = 52
C4_INDEX = 53

INDEX_START = 0
INDEX_END = 50 

def build_phrase_records(input_csv_records, phrase_dict):
    total_phrase_count = 0
    for i in range(0,len(input_csv_records)):
        if i%2 == 1:
            phrase_record = []
            for j in range(INDEX_MIN, INDEX_OUT+1):
                phrase_record.append(input_csv_records[i-1][j])
            
            for j in range(INDEX_MIN, INDEX_OUT+1):
                phrase_record.append(input_csv_records[i][j])

            for j in range(CONTEXT_MIN, CONTEXT_MAX+1):
                phrase_record.append(input_csv_records[i][j])
            
            phrase_dict[total_phrase_count] =  phrase_record
            total_phrase_count += 1

    return total_phrase_count


def make_vocab_index(total_phrase_count, phrase_dict, vocab_dict, reverse_vocab_dict):
    index_count = 0
    for i in range(0, total_phrase_count):
        if phrase_dict[i][W1_INDEX] not in vocab_dict:
            vocab_dict[phrase_dict[i][W1_INDEX] ] = index_count
            reverse_vocab_dict[index_count] = phrase_dict[i][W1_INDEX]
            index_count += 1
        
        if phrase_dict[i][W2_INDEX] not in vocab_dict:
            vocab_dict[phrase_dict[i][W2_INDEX] ] = index_count
            reverse_vocab_dict[index_count] = phrase_dict[i][W2_INDEX]
            index_count += 1
        
        if phrase_dict[i][C1_INDEX] not in vocab_dict:
            vocab_dict[phrase_dict[i][C1_INDEX] ] = index_count
            reverse_vocab_dict[index_count] = phrase_dict[i][C1_INDEX]
            index_count += 1
        
        if phrase_dict[i][C2_INDEX] not in vocab_dict:
            vocab_dict[phrase_dict[i][C2_INDEX] ] = index_count
            reverse_vocab_dict[index_count] = phrase_dict[i][C2_INDEX]
            index_count += 1
        
        if phrase_dict[i][C3_INDEX] not in vocab_dict:
            vocab_dict[phrase_dict[i][C3_INDEX] ] = index_count
            reverse_vocab_dict[index_count] = phrase_dict[i][C3_INDEX]
            index_count += 1
        
        if phrase_dict[i][C4_INDEX] not in vocab_dict:
            vocab_dict[phrase_dict[i][C4_INDEX] ] = index_count
            reverse_vocab_dict[index_count] = phrase_dict[i][C4_INDEX]
            index_count += 1
    
    return index_count

def make_valid_examples(total_phrase_count, phrase_dict, train_dict):
    train_count = 0
    for i in range(0, total_phrase_count):
        for k in range(0,4):
            phrase_record = []
            for j in range(INDEX_START, INDEX_END):
                phrase_record.append(phrase_dict[i][j])
            phrase_record.append(phrase_dict[i][INDEX_END + k])
            train_dict[train_count] = phrase_record
            train_count += 1
    
    return train_count

        

def generate_batch(batch_size, batch_no, adjusted_train_count, train_dict,\
                    vocab_dict,reverse_vocab_dict):

    offset = (batch_no % ( adjusted_train_count/batch_size ) * batch_size)

    batch_features1 = np.ndarray(shape=(batch_size, feature_size), dtype=np.int32)
    batch_features2 = np.ndarray(shape=(batch_size, feature_size), dtype=np.int32)
    batch_input_indices1 = np.ndarray(shape=(batch_size), dtype=np.int32)
    batch_input_indices2 = np.ndarray(shape=(batch_size), dtype=np.int32)
    batch_output_indices = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    for i in range(0,batch_size):
        #print i, i + offset,  train_dict[i + offset ][W1_INDEX],  train_dict[i + offset ][W2_INDEX], len(vocab_dict.keys()), vocab_dict['outstanding']
        batch_features1[i] = np.asarray(train_dict[i + offset ][W1_INDEX+1 : W2_INDEX])
        batch_features2[i] = np.asarray(train_dict[i + offset ][W2_INDEX+1 : C1_INDEX])
        batch_input_indices1[i] = np.asarray(vocab_dict [ train_dict[i + offset ][W1_INDEX] ])
        batch_input_indices2[i] = np.asarray(vocab_dict [ train_dict[i + offset ][W2_INDEX] ])
        batch_output_indices[i] = np.asarray(vocab_dict [ train_dict[i + offset ][INDEX_END] ])
    
    return batch_features1, batch_features2, batch_input_indices1, batch_input_indices2, batch_output_indices


#load data
vocab_dict = None
with open('vocab.pickle', 'rb') as handle:
    vocab_dict = pickle.load(handle)

reverse_vocab_dict = None
with open('reverse_vocab.pickle', 'rb') as handle:
    reverse_vocab_dict = pickle.load(handle)



phrase_dict = None
with open('phrase.pickle', 'rb') as handle:
    phrase_dict = pickle.load(handle)

train_dict = None
with open('train.pickle', 'rb') as handle:
    train_dict = pickle.load(handle)


np_embeddings = None
with open('embeddings_reviews.pickle', 'rb') as handle:
    np_embeddings = pickle.load(handle)


train_count = len(train_dict.keys())
adjusted_train_count = ( train_count/batch_size )* batch_size


vocabulary_size = len(vocab_dict.keys())
embedding_size = 300
batch_size = 300
feature_size = 24



num_steps = 10000

graph = tf.Graph()

with graph.as_default():

    train_input_feature_set1 = tf.placeholder(tf.float32, shape=[batch_size,feature_size])
    train_input_feature_set2 = tf.placeholder(tf.float32, shape=[batch_size, feature_size])

    train_input_indices_word1 = tf.placeholder(tf.int32, shape=[batch_size])
    train_input_indices_word2 = tf.placeholder(tf.int32, shape=[batch_size])

    train_labels_indices = tf.placeholder(tf.int32, shape=[batch_size, 1])

    #embedding_init_place = tf.placeholder(tf.float32, shape=(vocabulary_size, embedding_size))
    #embeddings = tf.Variable(embedding_init_place)

    with tf.device('/cpu:0'):

#        embeddings = tf.Variable(
#            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        embeddings = tf.Variable(np_embeddings, name="embeddings")

        feature_weights = tf.Variable(tf.random_uniform([feature_size, embedding_size], -1.0, 1.0))
        feature_bias = tf.Variable(tf.zeros([embedding_size]))

        embed1 = tf.nn.embedding_lookup(embeddings, train_input_indices_word1)
        embed2 = tf.nn.embedding_lookup(embeddings, train_input_indices_word2)

    # Construct the variables for the NCE loss aka outputweights, output bias
        nce_weights = tf.Variable(
                    tf.truncated_normal([vocabulary_size, embedding_size],
                                        stddev=1.0 / math.sqrt(embedding_size)))
    
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        l1 = tf.add(tf.matmul(train_input_feature_set1, feature_weights), feature_bias)
        l2 = tf.add(tf.matmul(train_input_feature_set2, feature_weights), feature_bias)


        final_input = tf.add(tf.mul(l1, embed1), tf.mul(l2, embed2))

    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels_indices,
                       inputs=final_input,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    init = tf.initialize_all_variables()


with tf.Session(graph=graph) as session:

    init.run()
    average_loss = 0

    for step in xrange(num_steps):
        batch_features1, batch_features2,\
        batch_input_indices1, batch_input_indices2,\
        batch_output_indices,\
            = generate_batch(batch_size, step, adjusted_train_count, train_dict,\
                    vocab_dict,reverse_vocab_dict)

        feed_dict = {train_input_feature_set1: batch_features1, \
                        train_input_feature_set2: batch_features2, \
                        train_input_indices_word1: batch_input_indices1, \
                        train_input_indices_word2: batch_input_indices2, \
                        train_labels_indices: batch_output_indices}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        print loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

    final_embeddings = normalized_embeddings.eval() 
    final_weights = feature_weights.eval()
    final_bias  = feature_bias.eval()


print "Train Completed"
