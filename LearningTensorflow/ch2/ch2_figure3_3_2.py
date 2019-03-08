#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 14:08:11 2018

@author: daiyucheng
"""

import tensorflow as tf

a = tf.constant(12, dtype=tf.float32)
b = tf.constant(23, dtype=tf.float32)

c = a * b
d = tf.sin(c)

e = d / b

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(e)
    
print(res)

print(tf.get_default_graph())

additional_grap = tf.Graph()

print(additional_grap)

print(a.graph)
print(a.graph is tf.get_default_graph())         # True
print(additional_grap is tf.get_default_graph()) # False