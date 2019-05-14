#!/usr/bin/env python
# encoding: utf-8

# 说明：　该程序是利用keras设计的一个包含N个隐藏层的神经网络。
#        其中, N主要由hide_layer_size参数决定
# 数据集： MNIST

import numpy as np
# import tensorflow.keras as keras
import keras
import keras.datasets.mnist as mnist
# import tensorflow.keras.datasets.mnist as mnist
from livelossplot.keras import PlotLossesCallback  # pip install livelossplot
import os

class MNISTFullyConnectedNetwork(object):
    """docstring for MNIST_FullyConnectedNetwork"""
    def __init__(self):
        self.model = keras.Sequential()
        self.plot_losses = PlotLossesCallback()
        self.save_model_img = 'image/fully_connected_network_model.png'
        self.save_model_file = 'model/fully_connected_network_model.h5'

    def load_design_train_same(self,  hide_layer_size=(512, 512)):
        # 加载数据
        (x_train, y_train), (x_test, y_test) = MNISTFullyConnectedNetwork.read()

        print(x_train.shape)
        print(y_train.shape)

        # 元组相加：　向隐藏层前后分别增加输入层和输出层节点数
        layer_size = (x_train.shape[1],) + hide_layer_size + (y_train.shape[1],)

        # 设计网络模型
        self.design(layer_size)

        # 训练模型
        self.train(x_train, y_train, x_test, y_test)

        # 保存模型
        self.save_model()

    # 加载数据,静态成员方法, Note: class method and cls
    @classmethod
    def read(cls, num_classes=10):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # 对数据进行归一化到0-1
        x_train /= 255
        x_test /= 255

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

        # 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        return (x_train, y_train), (x_test, y_test)

    # 定义基于全连接网络的mnist分类模型
    def design(self, layer_size):
        if len(layer_size) <= 2:
            print("Inappropriate parameter: lay_size, len(lay_size) should be larger than 2")
            return False

        # model.add(keras.layers.Dense(layer_size[1], input_shape=(layer_size[0],),
        # activation= keras.layers.Activation('relu')))
        self.model.add(keras.layers.Dense(layer_size[1], input_shape=(layer_size[0],)))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.Dropout(0.3))

        # 隐藏层
        for sz in layer_size[2:-1]:
            self.model.add(keras.layers.Dense(sz))
            self.model.add(keras.layers.Activation('relu'))
            self.model.add(keras.layers.Dropout(0.3))

        self.model.add(keras.layers.Dense(10))
        self.model.add(keras.layers.Activation('softmax'))

        self.model_info()
        return True

    def model_info(self, to_file=None):
        print(self.model.summary())

        if to_file == None:
            to_file = self.save_model_img

        keras.utils.plot_model(self.model, to_file)

    # 训练和评估已定义好的模型
    def train(self, x_train, y_train, x_test, y_test, batch_size=128, nb_epoch=10):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.RMSprop(),
                           metrics=['accuracy'])

        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=nb_epoch,
                       verbose=1,
                       validation_data=(x_test, y_test),
                       callbacks=[self.plot_losses])

        score = self.model.evaluate(x_test, y_test, verbose=0)

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
            filename = self.save_model_file

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


def run(retrain=True):
    fcn = MNISTFullyConnectedNetwork()
    if retrain:
        fcn.load_design_train_same((512,))  # 传入不同的隐藏层节点数(512, 512, 256)
    else:
        fcn.load_model()

    _, (x_test, y_test) = fcn.read()

    # 识别单个数据
    single_num = fcn.predict(x_test[1].reshape((1, 784)))
    print("single_num:", single_num)

    # 识别多个数据
    nums = fcn.predict(x_test[1:10])
    print("nums:", nums)


if __name__ == '__main__':  
    # 传入参数True表示重新训练模型,否False时,将从self.save_model_file中加载模型
    run(True)
