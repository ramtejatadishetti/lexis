import tensorflow as tf
import numpy as np

feature_size = 10
vocabulary_size = 10
embedding_size = 300
batch_size = 1
num_sampled = 64    # Number of negative examples to sample.

global_dict_for_indices = {}
count_of_tokens ={}

# helper function to build token dictionary
def make_indices(tokens):
    index_count = 0
    for i in range(0, len(tokens)):
        if tokens[i] in global_dict_for_indices:
            count_of_tokens[global_dict_for_indices[tokens[i]]] += 1

        else:
            global_dict_for_indices[tokens[i]] = index_count
            count_of_tokens[global_dict_for_indices[tokens[i]]] = 1
            index_count += 1




def make_batches(bigram_phrase, context_for_bigrams):

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)



def graph():

    train_input_feature_set1 = tf.placeholder(tf.int32, shape=[batch_size,feature_size])
    train_input_feature_set2 = tf.placeholder(tf.int32, shape=[batch_size, feature_size])

    train_input_indices_word1 = tf.placeholder(tf.int32, shape=[batch_size])
    train_input_indices_word2 = tf.placeholder(tf.int32, shape=[batch_size])

    train_labels_indices = tf.placeholder(tf.int32, shape=[batch_size, 1])

    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    feature_weights = tf.Variable(tf.random_uniform([feature_size, embedding_size], -1.0, 1.0))
    feature_bias = tf.Variable(tf.zeros([feature_size]))

    embed1 = tf.nn.embedding_lookup(embeddings, train_input_indices_word1)
    embed2 = tf.nn.embedding_lookup(embeddings, train_input_indices_word2)

    # Construct the variables for the NCE loss aka outputweights, output bias
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


    l1 = tf.add(tf.matmul(train_input_feature_set1, feature_weights), feature_bias)
    l2 = tf.add(tf.matmul(train_input_feature_set2, feature_weights), feature_bias)

    final_input = tf.add(tf.multiply(l1, embed1), tf.multiply(l2, embed2))

    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels_indices,
                       inputs=final_input,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm


