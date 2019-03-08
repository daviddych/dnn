#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 10:09:37 2018

@author: daiyucheng

Classifiying MNIST handwritten digits with softmax regression.
"""

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

DATA_DIR = '/home/daiyucheng/data/MNIST'
print(os.path.exists(DATA_DIR))

NUM_STEPS=1000
MINIBATCH_SIZE=100

data = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x,W)

# cross entropy: a natural choice when the model outputs class probabilities.
loss_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
# loss_cost = tf.losses.softmax_cross_entropy(y_true, y_pred)
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss_cost)

#correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32), name='accuracy')
#accuracy = tf.metrics.accuracy(labels = tf.argmax(y_true, 1), predictions=tf.argmax(y_pred, 1))
accuracy, acc_op = tf.metrics.accuracy(labels=tf.argmax(y_true, 1), predictions=tf.argmax(y_pred,1))

with tf.Session() as sess:
    print("Begin >>> Train")
    # Train
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    accs=[]
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x:batch_xs, y_true:batch_ys})
        
        # Test
        ans = sess.run(acc_op, feed_dict={x:data.test.images, y_true:data.test.labels})
        print("Accuracy:{:.4}%".format(ans * 100))
        accs.append(ans)
        
        
    print("End <<< Train")

# draw 
plt.plot(accs)
plt.ylabel("Accuracy")
plt.xlabel('iteraters')
plt.show()