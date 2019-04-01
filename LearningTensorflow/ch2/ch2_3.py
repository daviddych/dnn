#!/usr/bin/env python3
# coding:utf-8
import tensorflow as tf
import numpy as np

x_data = np.random.randn(5, 10)
w_data = np.random.randn(10, 2)

with tf.Graph().as_default():
    c1 = tf.constant(4, dtype=tf.int32, name='c')
    with tf.name_scope('prefix_name'):
        c2 = tf.constant(4, dtype=tf.int32, name='c')
        c3 = tf.constant(4, dtype=tf.int32, name='c')

    x = tf.placeholder(tf.float32, shape=(5, 10))
    w = tf.placeholder(tf.float32, shape=(10, 2))
    b = tf.fill((5, 2), -1.0)

    xwb = tf.matmul(x, w) + b
    s = tf.reduce_max(xwb)

    print(c1.name)
    print(c2.name)
    print(c3.name)

    init_v = tf.random_normal((1, 5), 0, 1)
    var = tf.Variable(init_v, name='var')
    print("pre run:\n{}\n{}".format(var.name, var))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        post_var = sess.run(var)
        resxwb_, res = sess.run((s, xwb), feed_dict={x: x_data, w: w_data})

    print("xwb: {}".format(resxwb_))
    print('s:{}'.format(res))
    print("post run:\n{}".format(post_var))
