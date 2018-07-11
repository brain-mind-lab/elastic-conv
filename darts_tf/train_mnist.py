# Performs architecture and weights search for MNIST classifier

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from itertools import count

import time
import datetime
import os

from model import model

from utils import plot_cells, _fprint

slim = tf.contrib.slim

LOG_DIR = os.path.join('mnist_search', '%s' % datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
CELL_IMAGES_DIR = os.path.join(LOG_DIR, 'cells')
LOG_FILE = os.path.join(LOG_DIR, 'log.txt')
print = _fprint(LOG_FILE)

if not os.path.exists(CELL_IMAGES_DIR):
    os.makedirs(CELL_IMAGES_DIR)
    
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
x_train = mnist.train.images # Returns np.array
y_train = np.asarray(mnist.train.labels, dtype=np.int32)
x_valid = mnist.test.images # Returns np.array
y_valid = np.asarray(mnist.test.labels, dtype=np.int32)

# Split train set in order to separately train weight and architecture
shuffle_indices = np.random.choice(np.arange(x_train.shape[0]), x_train.shape[0], replace=False)
x_train_w = x_train[shuffle_indices[:(x_train.shape[0] // 2)]]
y_train_w = y_train[shuffle_indices[:(x_train.shape[0] // 2)]]
x_train_a = x_train[shuffle_indices[(x_train.shape[0] // 2):]]
y_train_a = y_train[shuffle_indices[(x_train.shape[0] // 2):]]

BATCH_SIZE = 100

def next_batch(num, data, labels):
    idx = np.arange(0 , data.shape[0])
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]

    return data_shuffle, labels_shuffle


_images = tf.placeholder(tf.float32, (None, 784), name='images')
_labels = tf.placeholder(tf.int32, (None,), name='labels')

_onehot_labels = tf.one_hot(tf.expand_dims(_labels, axis=1), depth=10)


_net = tf.reshape(_images, (-1, 28, 28, 1))

# Stem
_net = slim.conv2d(_net, 32, (5, 5), stride=(2, 2), padding='VALID')

# Construct DARTS model
# alpha stores list of variables, responsible for architecture only
_net, alpha = model(_net, [32, 64, 64], [False, True, True], 1)

_net = slim.conv2d(_net, 256, (3, 3), padding='VALID', activation_fn=tf.nn.relu)

_net = tf.keras.layers.Flatten()(_net)

_net = tf.layers.dense(inputs=_net, units=1024, activation=tf.nn.relu)
_net = tf.layers.dropout(inputs=_net, rate=0.4)
_logits = tf.layers.dense(inputs=_net, units=10)

_classes = tf.nn.softmax(_logits)

#_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=_logits, labels=_onehot_labels))
_loss = tf.losses.sparse_softmax_cross_entropy(labels=_labels, logits=_logits)

alpha = alpha + tf.get_collection('alpha')
global_step = tf.Variable(0, trainable=False)
train_learning_rate = tf.train.exponential_decay(1e-4, global_step,
                                           100, 0.96, staircase=False)

w = [e for e in tf.trainable_variables() if not (e in alpha)]

# Define optimizer for first optimization objective (change layer weights for minimizing loss on train set)
_w_opt = (
    tf.train.MomentumOptimizer(learning_rate=1e-1, momentum=0.9).minimize(_loss, global_step=global_step, var_list=w)
)
# Define optimizer for second optimization objective (change architecture weights for minimizing loss on valid set)
_alpha_opt = (
    tf.train.AdamOptimizer(1e-1).minimize(_loss, global_step=global_step, var_list=alpha)
)

print('Model created')

with tf.Session() as sess:
    print('Session created')
    sess.run(tf.global_variables_initializer())
    print('Initialization done')

    start_time = time.time()
    for i in count():

        valid_x, valid_y = next_batch(BATCH_SIZE, x_valid, y_valid)

        # Validate
        valid_loss, labels = sess.run([_loss, _labels], feed_dict={
            _images: valid_x,
            _labels: valid_y
        })

        if i % 100 == 0:
            logits, labels = list(), list()
            for b in range(0, x_valid.shape[0], BATCH_SIZE):
                lo, la = sess.run([_classes, _labels], feed_dict={
                    _images: x_valid[b:b+BATCH_SIZE],
                    _labels: y_valid[b:b+BATCH_SIZE],
                })
                logits.append(lo.argmax(axis=1))
                labels.append(la)
            logits = np.concatenate(logits, axis=0)
            labels = np.concatenate(labels, axis=0)

            print('Accuracy: %.2f' % ((logits == labels).mean() * 100) + '%')
            
        if i % 100 == 0:
            alpha_normal, alpha_normal_econv, alpha_reduction, alpha_reduction_econv = sess.run(alpha)
            cell_file = os.path.join(CELL_IMAGES_DIR, '%06d' % i)
            plot_cells(alpha_normal, alpha_normal_econv, alpha_reduction, alpha_reduction_econv, cell_file)
            
        train_w_x, train_w_y = next_batch(BATCH_SIZE, x_train_w, y_train_w)
        train_a_x, train_a_y = next_batch(BATCH_SIZE, x_train_a, y_train_a)

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

        if i % 10 == 0:
            print('Step #%i: train - %.4f valid - %.4f time (s) - %.2f' % (i, train_loss, valid_loss, (time.time() - start_time)))
            start_time = time.time()
