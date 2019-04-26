#!/usr/bin/env python
# encoding: utf-8

# 说明：　该程序是利用keras实现了一个包含2个卷积+2个全连接+一个softmax的神经网络。
#
# 数据集： MNIST

import numpy as np
import tensorflow.keras as keras
import tensorflow.keras.datasets.mnist as mnist
from   livelossplot.keras import PlotLossesCallback  # pip install livelossplot
import time
import os

class ConvolutionalNeuralNetwork(object):
    """docstring for MNIST_ConvolutionalNeuralNetwork"""
    def __init__(self):
        self.model = keras.Sequential()
        self.plot_losses = PlotLossesCallback()
        self.save_model_img = 'image/convolutional_neural_network_model_mnist.png'
        self.save_model_file = 'model/convolutional_neural_network_model_mnist.h5'

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

        # 将图片从三维矩阵reshape为四维矩阵, keras输入数据有两种格式,一种是通道数放在前面,一种是通道数放在后面
        if keras.backend.image_data_format == 'channels_first':
            self.input_shape = (1, x_train.shape[1], x_train.shape[2])
            x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
            x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
        else:
            self.input_shape = (x_train.shape[1], x_train.shape[2], 1)
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        # 转成float32使后面的计算更精确
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # 对数据进行归一化到0-1
        x_train /= 255
        x_test /= 255

        # 将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        return (x_train, y_train), (x_test, y_test)

    # 定义基于Keras的简单卷积网络分类模型
    def design(self):
        # if len(layer_size) <= 2:
        #     print("Inappropriate parameter: lay_size, len(lay_size) should be larger than 2")
        #     return False

        # 第一层()
        conv1 = keras.layers.Convolution2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=self.input_shape)
        self.model.add(conv1)
        self.model.add(keras.layers.Activation('relu'))

        # 第二层
        self.model.add(keras.layers.Convolution2D(64, (3, 3), padding='same'))
        self.model.add(keras.layers.Activation('relu'))
        self.model.add(keras.layers.MaxPooling2D())
        self.model.add(keras.layers.Dropout(0.25))  # Dense()的前面要减少连接点，防止过拟合，故通常要Dropout层或池化层

        # 第三层(), 首先展平所有像素,eg.[14*14*64] --> [12544]
        self.model.add(keras.layers.Flatten())      # Dense()层的输入通常是2D张量，故应使用Flatten层或全局平均池化
        self.model.add(keras.layers.Dense(128))     # 对所有像素使用全连接,输出为128维
        self.model.add(keras.layers.Activation('relu'))  # Dense()层的后面通常要加非线性化函数
        self.model.add(keras.layers.Dropout(0.5))        # 对输入采用0.5概率的Dropout

        # 第四层
        self.model.add(keras.layers.Dense(10, activation='softmax')) # 分类网络最后层,通常是softmax

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['accuracy'])

        self.model_info(to_file = self.save_model_img)

        return True

    def model_info(self, to_file=None):
        ''' 打印网络结构 + 保存为图像文件 '''
        print(self.model.summary())
        if to_file == None:
            to_file = self.save_model_img

        keras.utils.plot_model(self.model, to_file)

    # 训练和评估已定义好的模型
    def train(self, x_train, y_train, x_test, y_test, batch_size=128, nb_epoch=10):
        # 将compile上移

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
            filename = self.save_model_file #self.str_time() + '.h5'

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
    fcn = ConvolutionalNeuralNetwork()
    if retrain:
        fcn.load_design_train_same()
    else:
        fcn.load_model()

    _, (x_test, y_test) = fcn.read()

    # 识别单个数据, 切记首先要reshape数据
    single_num = fcn.predict(x_test[1].reshape((1, 28, 28, 1)))
    print("single_num:", single_num)

    # 识别多个数据
    nums = fcn.predict(x_test[1:10])
    print("nums:", nums)


if __name__ == '__main__':
    # 传入参数True表示重新训练模型,否False时,将从self.save_model_file中加载模型
    run(True)
