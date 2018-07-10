# Performs architecture and weights search for MNIST classifier

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from itertools import count

slim = tf.contrib.slim
import time

from model import model

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
x_train = mnist.train.images # Returns np.array
y_train = np.asarray(mnist.train.labels, dtype=np.int32)
x_valid = mnist.test.images # Returns np.array
y_valid = np.asarray(mnist.test.labels, dtype=np.int32)

BATCH_SIZE = 32

def next_batch(num, data, labels):
    idx = np.arange(0 , data.shape[0])
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


_images = tf.placeholder(tf.float32, (None, 784), name='images')
_labels = tf.placeholder(tf.uint8, (None,), name='labels')

_onehot_labels = tf.one_hot(tf.expand_dims(_labels, axis=1), depth=10)


_net = tf.reshape(_images, (-1, 28, 28, 1))

# Stem
_net = slim.conv2d(_net, 4, (5, 5), stride=(2, 2), padding='VALID')

# Construct DARTS model
# alpha stores list of variables, responsible for architecture only
_net, alpha = model(_net, [4, 8, 8, 16, 16], [False, True, False, True, False], 5)

_net = slim.conv2d(_net, 10, (3, 3), padding='VALID', activation_fn=tf.nn.softmax)

_logits = tf.keras.layers.Flatten()(_net)

#_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=_logits, labels=_onehot_labels))
_loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(_labels, tf.int32), logits=_logits)

alpha = alpha + tf.get_collection('alpha')
global_step = tf.Variable(0, trainable=False)
train_learning_rate = tf.train.exponential_decay(1e-4, global_step,
                                           100, 0.96, staircase=False)

w = [e for e in tf.trainable_variables() if not (e in alpha)]

# Define optimizer for first optimization objective (change layer weights for minimizing loss on train set)
_w_opt = (
    tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9).minimize(_loss, global_step=global_step, var_list=w)
)
# Define optimizer for second optimization objective (change architecture weights for minimizing loss on valid set)
_alpha_opt = (
    tf.train.AdamOptimizer(0.5).minimize(_loss, global_step=global_step, var_list=alpha)
)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    start_time = time.time()
    for i in count():
        train_w_x, train_w_y = next_batch(BATCH_SIZE, x_train, y_train)
        train_a_x, train_a_y = next_batch(BATCH_SIZE, x_train, y_train)

        valid_x, valid_y = next_batch(BATCH_SIZE, x_valid, y_valid)

        # Train first optimization objective
        train_loss, _ = sess.run([_loss, _w_opt], feed_dict={
            _images: train_w_x,
            _labels: train_w_y
        })

        # Train second optimization objective
        sess.run(_alpha_opt, feed_dict={
            _images: train_a_x,
            _labels: train_a_y
        })

        # Validate
        valid_loss = sess.run(_loss, feed_dict={
            _images: valid_x,
            _labels: valid_y
        })

        if i % 10 == 0:
            print('Step #%i: train - %.4f valid - %.4f time (s) - %.2f' % (i, train_loss, valid_loss, (time.time() - start_time)))
            start_time = time.time()