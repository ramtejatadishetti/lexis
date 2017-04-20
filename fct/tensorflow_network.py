import tensorflow as tf
import numpy as np

init_a = np.zeros(3)+1
init_b = np.zeros(3)+2



a = tf.placeholder("float", shape=[3])
b = tf.placeholder("float", shape=[3])

y = tf.multiply(a,b)

with tf.Session() as sess:
    print("%s should equal 2.0" % sess.run(y, feed_dict={a: init_a, b: init_b}))