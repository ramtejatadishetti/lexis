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

num_sampled = 300    # Number of negative examples to sample.

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


embedding_size = 300
batch_size = 300
RANGE_MIN = 0
feature_size = 24

shuffle_seed = 0.5


#import math
#val = float('nan')
#val
#if math.isnan(val):
#    print('Detected NaN')
#    import pdb; pdb.set_trace()

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


def check_in_blacklist(word):
    if word == 'dreadful':
        return 1
    elif word == 'duvets':
        return 1
    elif word == 'scuba':
        return 1
    elif word == 'swift':
        return 1
    elif word == 'para':
        return 1
    elif word == 'dormir':
        return 1
    elif word == 'temples':
        return 1
    elif word == 'veramente':
        return 1
    elif word == 'comodissimo':
        return 1
    elif word == 'purple':
        return 1
    elif word == 'panties':
        return 1


    else:
        return 0

def generate_batch(batch_size, batch_no, adjusted_train_count, train_dict,\
                    vocab_dict,reverse_vocab_dict, random_np_array):


    flag = 0;    
#    if (batch_no == 26):
#        flag = 1

#    if (batch_no > 24):
#        batch_no += 1

#    if (batch_no > 48):
#        batch_no += 2

#    if (batch_no > 82):
#        batch_no += 4


    offset = ( (batch_no % ( adjusted_train_count/batch_size )) * batch_size)

    batch_features1 = np.ndarray(shape=(batch_size, feature_size), dtype=np.int32)
    batch_features2 = np.ndarray(shape=(batch_size, feature_size), dtype=np.int32)
    batch_input_indices1 = np.ndarray(shape=(batch_size), dtype=np.int32)
    batch_input_indices2 = np.ndarray(shape=(batch_size), dtype=np.int32)
    batch_output_indices = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    for i in range(0,batch_size):
        word1 = train_dict[ random_np_array[i + offset] ][W1_INDEX] 
        word2 = train_dict[ random_np_array[i + offset] ][W2_INDEX]
        if( (check_in_blacklist(word1) == 1) or (check_in_blacklist(word2) == 1) ):
            batch_features1[i] = np.asarray(train_dict[ random_np_array[ 0 ] ][W1_INDEX+1 : W2_INDEX])
            batch_features2[i] = np.asarray(train_dict[ random_np_array[ 0 ] ][W2_INDEX+1 : C1_INDEX])
            batch_input_indices1[i] = np.asarray(vocab_dict [ train_dict[ random_np_array[ 0] ][W1_INDEX] ])
            batch_input_indices2[i] = np.asarray(vocab_dict [ train_dict[ random_np_array[ 0] ][W2_INDEX] ])
            batch_output_indices[i] = np.asarray(vocab_dict [ train_dict[ random_np_array[ 0] ][INDEX_END] ])
#            print train_dict[ random_np_array[ 0] ][W1_INDEX] +"-"+ train_dict[ random_np_array[ 0] ][W2_INDEX] , train_dict[ random_np_array[ 0] ][INDEX_END]

    
            
        else:
        #print i, i + offset,  train_dict[i + offset ][W1_INDEX],  train_dict[i + offset ][W2_INDEX], len(vocab_dict.keys()), vocab_dict['outstanding']
            batch_features1[i] = np.asarray(train_dict[ random_np_array[i + offset] ][W1_INDEX+1 : W2_INDEX])
            batch_features2[i] = np.asarray(train_dict[ random_np_array[i + offset] ][W2_INDEX+1 : C1_INDEX])
            batch_input_indices1[i] = np.asarray(vocab_dict [ train_dict[ random_np_array[i + offset] ][W1_INDEX] ])
            batch_input_indices2[i] = np.asarray(vocab_dict [ train_dict[ random_np_array[i + offset] ][W2_INDEX] ])
            batch_output_indices[i] = np.asarray(vocab_dict [ train_dict[ random_np_array[i + offset] ][INDEX_END] ])



#            print train_dict[ random_np_array[i + offset] ][W1_INDEX] +"-"+ train_dict[ random_np_array[i + offset] ][W2_INDEX] , train_dict[ random_np_array[i + offset] ][INDEX_END]
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
adjusted_train_count = ( train_count/2000 )* 2000


vocabulary_size = len(vocab_dict.keys())

print ('NUMBER OF TRAINING INPUTS:', adjusted_train_count)

print ('VOCABULARY SIZE: ', vocabulary_size)


#num_sampled = vocabulary_size -1;

random_np_array = list(xrange(adjusted_train_count))


random.shuffle(random_np_array, lambda: 0.5)
print random_np_array[0], random_np_array[1], random_np_array[2], random_np_array[adjusted_train_count-1]

num_steps = 1000000

graph = tf.Graph()

with graph.as_default():

    train_input_feature_set1 = tf.placeholder(tf.float32, shape=[batch_size,feature_size])
    train_input_feature_set2 = tf.placeholder(tf.float32, shape=[batch_size, feature_size])

    train_input_indices_word1 = tf.placeholder(tf.int32, shape=[batch_size])
    train_input_indices_word2 = tf.placeholder(tf.int32, shape=[batch_size])

    train_labels_indices = tf.placeholder(tf.int32, shape=[batch_size, 1])

    embeddings = tf.placeholder( tf.float32, shape=[vocabulary_size, 300])

#    preinit = tf.constant(np_embeddings, shape=[vocabulary_size, 300], dtype=tf.float32)

    #embedding_init_place = tf.placeholder(tf.float32, shape=(vocabulary_size, embedding_size))
    #embeddings = tf.Variable(embedding_init_place)

    with tf.device('/cpu:0'):

#        embeddings = tf.Variable(
#            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))


        #embeddings = tf.get_variable('embeddings', initializer=preinit, validate_shape=False)

        #embeddings = tf.Variable(np_embeddings, name="embeddings")

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
        tf.nn.sampled_softmax_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels_indices,
                       inputs=final_input,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    optimizer = tf.train.AdamOptimizer(0.000001).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    init = tf.initialize_all_variables()


print ("Total input_count",  adjusted_train_count)


with tf.Session(graph=graph) as session:

    init.run()
    average_loss = 0

    for step in range(RANGE_MIN, num_steps):
        batch_features1, batch_features2,\
        batch_input_indices1, batch_input_indices2,\
        batch_output_indices,\
            = generate_batch(batch_size, step, adjusted_train_count, train_dict,\
                    vocab_dict,reverse_vocab_dict, random_np_array)

#        print (batch_features1[0])
#        print ("Current_offset", step*300)
        feed_dict = {train_input_feature_set1: batch_features1, \
                        train_input_feature_set2: batch_features2, \
                        train_input_indices_word1: batch_input_indices1, \
                        train_input_indices_word2: batch_input_indices2, \
                        train_labels_indices: batch_output_indices, \
                        embeddings: np_embeddings}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        print( step, "Loss", loss_val)

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
