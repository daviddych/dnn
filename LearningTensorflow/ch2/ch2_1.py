# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 23:25:13 2018

@author: daiyucheng
"""
import tensorflow as tf

print(tf.__version__)

h = tf.constant("hello")
w = tf.constant(" World!")
hw = h + w

with tf.Session() as sess:
    ans = sess.run(hw)
    
    print(ans)