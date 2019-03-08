#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 14:03:47 2018

@author: daiyucheng
"""

import tensorflow as tf

a = tf.constant(12)
b = tf.constant(23)

c = a * b
d = a + b

e = c - d
f = c + d

g = e / f

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(g)

print(res)