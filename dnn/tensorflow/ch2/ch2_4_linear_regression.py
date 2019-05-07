#!/usr/bin/envs python3
#coding:-utf8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# generate data
x_data = np.random.randn(2000,3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2
noise = np.random.randn(1,2000)*0.01
y_data = np.matmul(w_real, x_data.T) + b_real + noise

print(y_data)

# draw raw data
plt.plot(x_data, y_data, 'r*')
plt.ylabel("Accuracy")
plt.xlabel('iteraters')
plt.show()

def draw_data():
    plt.plot(y_data)

NUM_STEP = 20
wb_=[]
with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)

    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0.0, 0.0, 0.0]], dtype = tf.float32, name = 'Weights')
        b = tf.Variable(0.0, dtype = tf.float32, name = 'bias')

        y_pred = tf.matmul(w, tf.transpose(x)) + b

    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optermizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optermizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEP+1):
            sess.run(train, {x:x_data, y_true:y_data})
            if step % 5 == 0:
                wb = sess.run([w,b])
                print(step, wb)
                wb_.append(wb)