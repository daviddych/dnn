#!/usr/bin/env python
#coding:utf-8

import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data


def load_data():
    ''' 
    : load data return:
    '''
    mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
    # 属性设置
    trainNum = 55000
    testNum = 10000
    trainSize = 500
    testSize = 5
    k = 4

    # data 分解
    trainIndex = np.random.choice(trainNum, trainSize, replace=False)
    testIndex = np.random.choice(testNum, testSize, replace=False)
    trainData = mnist.train.images[trainIndex]
    trainLabel = mnist.train.labels[trainIndex]
    testData = mnist.test.images[testIndex]
    testLabel = mnist.test.labels[testIndex]
    print(trainData.shape)
    print(trainLabel.shape)
    print(testData.shape)
    print(testLabel.shape)
    print(testLabel)

    return  trainData, trainLabel, testData, testLabel


if __name__ == '__main__':
        trainData, trainLabel, testData, testLabel = load_data()

        trainDataInput = tf.placeholder(shape=[None, 784], dtype=tf.float32)
        trainLabelInput = tf.placeholder(shape=[None, 10], dtype=tf.float32)
        testDataInput = tf.placeholder(shape=[None, 784], dtype=tf.float32)
        testLabelInput = tf.placeholder(shape=[None, 10], dtype=tf.float32)
        f1 = tf.expand_dims(testDataInput, 1) # 维度扩展
        f2 = tf.subtract(trainDataInput, f1)  # 784 sum(784)
        f3 = tf.reduce_sum(tf.abs(f2), reduction_indices=2) # 完成数据累加
        f4 = tf.negative(f3)
        f5, f6 = tf.nn.top_k(f4, k =4) # 选取f4中最大的四个值
        f7 = tf.gather(trainDataInput, f6)

        with tf.Session() as sess:
            p1 = sess.run(fetches=f1, feed_dict={testDataInput:testData[0:5]}) # (5, 1, 784)
            print('p1 = ', p1.shape)

            tDI = sess.run(fetches=trainDataInput, feed_dict={trainDataInput:trainData})
            print("tDI = ", tDI.shape)
            p2 = sess.run(fetches=f2, feed_dict={trainDataInput:trainData, testDataInput:testData[0:5]})
            print("trainDataInput = ", trainData.shape)
            print("testData = ", testData.shape)
            print("p2 = ", p2.shape) # (5, 500, 784)

            p3 = sess.run(fetches=f3, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
            print("p3 = ", p3.shape) # (5, 500)
            print("p3[0,0] = ", p3[0,0])

            p4 = sess.run(fetches=f4, feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
            print("p4 = ", p4.shape)  # (5, 500)
            print("p4[0,0] = ", p4[0, 0])

            p5, p6 = sess.run(fetches=(f5,f6), feed_dict={trainDataInput: trainData, testDataInput: testData[0:5]})
            print("p5 = ", p5.shape)  # (5, 500)
            print("p6 = ", p6.shape)  # (5, 500)
            print("p5[0,0] = ", p5[0, 0])
            print("p6[0,0] = ", p6[0, 0])

            print("p5 = ", p5) # 每一张图片分别对应的四张最近图片的距离
            print("p6 = ", p6) # 每张图片分别对应的四张最近图片的索引

            p7 = sess.run(fetches=f7, feed_dict={trainDataInput:trainData, testDataInput:testData[0:5], trainLabelInput:trainLabel})
            print('p7 = ', p7)