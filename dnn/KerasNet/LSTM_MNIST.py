#!/usr/bin/env python
# encoding: utf-8

# 说明：　该程序是利用keras实现了一个LSTM+1个全连接+一个softmax的神经网络。
#
# 数据集： MNIST

import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.datasets.mnist as mnist
from   livelossplot.keras import PlotLossesCallback  # pip install livelossplot
import time
import os

class LongShortTermMemory(object):
    """docstring for MNIST_LongShortTermMemory"""
    def __init__(self):
        self.model = keras.Sequential()
        self.plot_losses = PlotLossesCallback()
        self.save_model_img = 'image/LSTM_model_mnist.png'
        self.save_model_file = 'model/LSTM_model_mnist.h5'

        # LSTM的输出维度
        self.nb_lstm_outputs = 30

    def load_design_train_same(self):
        # 加载数据
        (x_train, y_train), (x_test, y_test) = self.read()

        print(x_train.shape)
        print(y_train.shape)

        # 设计网络模型
        self.design()

        # 训练模型
        self.train(x_train, y_train, x_test, y_test, 128, 5)

        # 保存模型
        self.save_model(self.save_model_file)

    # 加载数据
    def read(self, num_classes=10):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # 将图片reshape为三维矩阵
        nb_time_steps = x_train.shape[1]
        dim_input_vector = x_train.shape[2]
        self.input_shape = (nb_time_steps, dim_input_vector)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # 对数据进行归一化到0-1
        x_train /= 255
        x_test /= 255

        # 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        return (x_train, y_train), (x_test, y_test)

    # 定义基于全连接网络的mnist分类模型
    def design(self):
        # 第一层()
        self.model.add(keras.layers.LSTM(self.nb_lstm_outputs, input_shape=self.input_shape))  # 第一个参数是输出维度，第二个参数是输入图片的维度

        # 第二层
        self.model.add(keras.layers.Dense(10, activation='softmax', kernel_initializer=keras.initializers.random_normal(0.01)))

        # 查看网络结构
        self.model_info(to_file = self.save_model_img)

        return True

    # 查看和打印模型结构
    def model_info(self, to_file=None):
        print(self.model.summary())
        if to_file == None:
            to_file = self.save_model_img

        keras.utils.plot_model(self.model, to_file)

    # 训练和评估已定义好的模型
    def train(self, x_train, y_train, x_test, y_test, batch_size=128, nb_epoch=10):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=nb_epoch,
                       verbose=1,
                       shuffle=True,
                       validation_data=(x_test, y_test),
                       callbacks=[self.plot_losses])

        score = self.model.evaluate(x_test, y_test, verbose=1)

        print("Test score: {}", score[0])
        print("Test accuracy: {}", score[1])

    # 根据模型获取预测结果，为了节约计算内存，也是分组（batch）加载到内存
    def predict(self, x, batch_size=128, verbose=0):
        result = self.model.predict(x, batch_size=batch_size, verbose=verbose)

        # axis=1表示按行 取最大值   如果axis=0表示按列 取最大值 axis=None表示全部
        result_max = np.argmax(result, axis=1)

        return result_max

    # 保存训练好的模型
    def save_model(self, filename=None):
        if filename == None:
            filename = self.str_time() + '.h5'

        self.model.save(filename)

    # 加载已经训练好的模型
    def load_model(self, filename=None):
        if filename == None:
            filename = self.save_model_file

        if os.path.exists(filename) == False:
            print("Cannot find: ", filename)
            exit()

        self.model = keras.models.load_model(filename)
        self.model_info()

    # 获取系统时间字符串
    @classmethod
    def str_time(self):
        return  time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

def run(retrain=True):
    fcn = LongShortTermMemory()
    if retrain:
        fcn.load_design_train_same()  # 传入不同的隐藏层节点数(512, 512, 256)
    else:
        fcn.load_model()

    _, (x_test, y_test) = fcn.read()

    # 识别单个数据, 切记首先要reshape数据
    single_num = fcn.predict(x_test[1].reshape((1, x_test.shape[1], x_test.shape[2])))
    print("single_num:", single_num)

    # 识别多个数据
    nums = fcn.predict(x_test[1:10])
    print("nums:", nums)


if __name__ == '__main__':
    # 传入参数True表示重新训练模型,否False时,将从model/mnist-fully-connected-network.h5中加载模型
    run(False)
