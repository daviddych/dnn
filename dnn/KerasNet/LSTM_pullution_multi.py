#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:39:50 2019

@author: dyc
利用LSTMdui对环境质量时序数据进行分析预测, 训练模型,

相关说明:
    在train()内self.model.fit的回调参数中分别指定self.tbCallBack, self.plot_losses或self.history可体验不同数据可视化
    其中,取self.tbCallBack时,可以利用TensorBoard:
        首先, 激活tensorflow环境, 切换当前路径到该文件所处文件夹, 并运行该文件, 开始训练
        其次, 使用命令: tensorboard --logdir Graph, 会在命令行中显示类似 TensorBoard 1.11.0 at http://dyc:6006
        最后, 在浏览器中输入:  http://dyc:6006, 便可以看到数据可视化结果
"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
import tensorflow
from livelossplot.keras import PlotLossesCallback
import time
import os

from datetime import datetime


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


class History(tensorflow.keras.callbacks.Callback):
    """ 定义History类保存训练过程中的loss和acc信息 """

    def __init__(self):
        """ 创建字典盛放loss 和 acc """
        pass

    # 重写父类方法
    def on_batch_end(self, batch, logs={}):
        # 每个batch完成后向字典中追加loss和acc
        pass

    # 重写父类方法
    def on_train_end(self, logs=None):
        pass

    # 重写父类方法
    def on_epoch_end(self, batch, logs={}):
        print(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))

class SeriesLSTM(object):
    ''' 对时序数据利用LSTM方法预测 '''

    def __init__(self, n_in=1, n_out=1, dropnan=True):
        ''' 依据前n_in个时间, 预测后n_out个时间 '''
        self.model = tensorflow.keras.Sequential()

        self.plot_losses = PlotLossesCallback()
        self.history = History()
        self.tbCallBack = tensorflow.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        self.save_model_img = 'image/series_lstm_multi_class.png'
        self.save_model_file = 'image/series_lstm_multi_class.h5'  # self.model.save("stock_lstm.h5")
        self.n_hours = 3

    # define the function
    def draw_history(self, hist):
        ''' 绘制模型fit返回值history '''
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        acc = hist.history['acc']
        val_acc = hist.history['val_acc']

        # make a figure
        fig = plt.figure(figsize=(8, 4))
        # subplot loss
        ax1 = fig.add_subplot(121)
        if loss is not None:
            ax1.plot(loss, label='train_loss')
        if val_loss is not None:
            ax1.plot(val_loss, label='val_loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss on Training and Validation Data')
        ax1.legend()
        # subplot acc
        ax2 = fig.add_subplot(122)
        if acc is not None:
            ax2.plot(acc, label='train_acc')
        if val_acc is not None:
            ax2.plot(val_acc, label='val_acc')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy  on Training and Validation Data')
        ax2.legend()
        plt.tight_layout()
        plt.show()

    def read(self, csvfile = 'data/raw.csv'):
        ''' 加载数据,预处理,单位化 '''
        dataset = self.data_preparation(srcfile=csvfile, dstfile='data/pollution.csv')
        #self.plots_series(dataset=dataset)
        reframed, self.scaler = self.normalized_features(data='data/pollution.csv')

        self.n_features = int(reframed.shape[1] / (self.n_hours + 1)) # =8

        # split into train and test sets
        self.train_X, self.train_y, self.test_X, self.test_y = self.split_train_test(reframed, train_vs_test=0.8)
        return self.train_X, self.train_y, self.test_X, self.test_y

    def load_design_train_save(self, datafile='data/raw.csv'):
        ''' 加载数据,设计模型,训练和保存模型 '''
        self.train_X, self.train_y, self.test_X, self.test_y = self.read(datafile)
        print('x_train.shape={}, y_train.shape={}'.format(self.train_X.shape, self.train_y.shape))

        # 设计网络模型
        self.model = self.design()

        # 训练网络模型
        self.train(self.train_X, self.train_y, self.test_X, self.test_y) # train_X, train_y, test_X, test_y

        # 保存网络模型
        self.save_model(self.save_model_file)

    def data_preparation(self, srcfile='data/raw.csv', dstfile='data/pollution.csv'):
        ''' 数据预处理, 对于直接下载的raw.csv文件, 重组datatime, 删除第一列No, 重命名列名, 0填充数据缺失, 剔除前24行 '''
        dataset = read_csv(srcfile, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
        dataset.drop('No', axis=1, inplace=True)
        # manually specify column names
        dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
        dataset.index.name = 'date'
        # mark all NA values with 0
        dataset['pollution'].fillna(0, inplace=True)
        # drop the first 24 hours
        dataset = dataset[24:]
        # summarize first 5 rows
        print(dataset.head(5))
        # save to file
        dataset.to_csv(dstfile)
        
        self.n_features = dataset.values.shape[1]

        return dataset

    # convert series to supervised learning
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        '''Shift all the observations down by one time step by inserting one new row at the top.Because
        the new row has no data, we can use NaN to represent “no data”. '''
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))  # DataFrame中的所有列下移i行, 顶部填充NaN
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))  # DataFrame中的所有列上移i行, 尾部填充NaN
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def split_train_test(self, reframed, train_vs_test=0.8):
        ''' 将数据划分为训练集和测试集, train_vs_test表示训练集占整个数据集部分的比例. '''
        values = reframed.values
        n_train_hours = int(values.shape[0] * train_vs_test)
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]
        # split into input and outputs
        n_obs = self.n_hours * self.n_features
        train_X, train_y = train[:, :n_obs], train[:, -self.n_features] 
        test_X, test_y = test[:, :n_obs], test[:, -self.n_features]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], self.n_hours, self.n_features))
        test_X = test_X.reshape((test_X.shape[0], self.n_hours, self.n_features))
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        return train_X, train_y, test_X, test_y

    def plots_series(self, dataset='data/pollution.csv'):  # file
        ''' 输入参数只能是文件名或DataFrame类型变量, 进行绘图 '''
        if isinstance(dataset, DataFrame):
            values = dataset.values
        elif type(dataset) == type("filename"):
            # load dataset
            dataset = read_csv(dataset, header=0, index_col=0)
            values = dataset.values
        else:
            print("输入参数只能是文件名或DataFrame类型变量")

        # specify columns to plot
        groups = [0, 1, 2, 3, 5, 6, 7]
        i = 1
        # plot each column
        pyplot.figure()
        for group in groups:
            pyplot.subplot(len(groups), 1, i)
            pyplot.plot(values[:, group])
            pyplot.title(dataset.columns[group], y=0.5, loc='right')
            i += 1
        pyplot.show()

    def normalized_features(self, data):
        ''' 传入DataFrame 或者csv文件, 标准化标签，将非数据列标签值统一转换成range(标签值个数-1)范围内数值型, 将数据重构适应于监督学习数据'''
        values = data.values if type(data) is DataFrame else read_csv(data, header=0, index_col=0).values
        encoder = LabelEncoder()
        values[:, 4] = encoder.fit_transform(values[:, 4])
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, self.n_hours, 1)
        print("reframed.shape = {}".format(reframed.shape))
        
        # drop columns we don't want to predict
        #reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
        #print("reframed.head():\n", reframed.head())

        return reframed, scaler

    # 定义基于Keras的简单卷积网络分类模型
    def design(self):
        ''' 设计lstm网络 '''
        self.model = Sequential()

        self.model.add(LSTM(50, input_shape=(self.train_X.shape[1], self.train_X.shape[2]), return_sequences=True))
        self.model.add(keras.layers.Dropout(0.4))
        self.model.add(LSTM(30))
        self.model.add(keras.layers.Dropout(0.25))

        self.model.add(Dense(1))
        self.model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])  # 添加: metrics=['accuracy'] 在fit模型返回值中就会有history['acc'] 和 history['val_acc']

        self.model_info(to_file=self.save_model_img)

        return self.model

    def model_info(self, to_file=None):
        ''' 打印网络结构, 保存为图像文件 '''
        print(self.model.summary())
        if to_file == None:
            to_file = self.save_model_img

        keras.utils.plot_model(self.model, to_file)

    # 获取系统时间字符串
    @classmethod
    def str_time(self):
        return  time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    def print_epoch(self):
        print(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))

    def train(self, train_X, train_y, test_X, test_y):
        # fit network
        hist = self.model.fit(train_X, train_y, epochs=200, batch_size=32, validation_data=(test_X, test_y), verbose=2,
                              shuffle=False, callbacks=[self.tbCallBack]) # self.plot_losses  , self.history

        score = self.model.evaluate(test_X, test_y, verbose=0)
        print("Test score: {}", score[0])
        print("Test accuracy: {}", score[1])

        self.draw_history(hist)
        return self.model

    # 根据模型获取预测结果，为了节约计算内存，也是分组（batch）加载到内存
    def predict(self, x, batch_size=128, verbose=0):
        result = self.model.predict(x, batch_size=batch_size, verbose=verbose)

        # axis=1表示按行 取最大值   如果axis=0表示按列 取最大值 axis=None表示全部
        return result

    def Evaluate_Model(self, test_X, test_y):
        ''' Evaluate the model '''
        # make a prediction: reconstruct the rows with 8 columns suitable for
        # reversing the scaling operation to get the y and yhat back into the original scale
        # so that we can calculate the RMSE.
        yhat = self.model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], self.n_hours*self.n_features))
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, -(self.n_features-1):]), axis=1)
        inv_yhat = self.scaler.inverse_transform(inv_yhat) # 上面scaler.fit_transform(values)时是8列,所以,此处不许先构造回8列
        inv_yhat = inv_yhat[:, 0]
        # The gist of the change is that we concatenate the y or yhat column with the last 7 features of the test
        # dataset in order to inverse the scaling

        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, -(self.n_features-1):]), axis=1)
        inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y[1:], inv_yhat[:-1]))
        print('Test RMSE: %.3f' % rmse)


    # 保存训练好的模型
    def save_model(self, filename=None):
        if filename == None:
            filename = self.save_model_file  # self.str_time() + '.h5'

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


def run():
    slstm = SeriesLSTM()
    slstm.load_design_train_save(datafile='data/raw.csv')
    slstm.Evaluate_Model(slstm.test_X, slstm.test_y)

def run_loadmodel_predict():
    slstm = SeriesLSTM()
    slstm.read()
    slstm.load_model()
    result = slstm.predict(slstm.test_X[-4:-1,:])
    print(result.shape)
    result = result.reshape((result.shape[0],))
    print("result: ", result)
    print("x[-1,:]:", slstm.test_y[-3:])

    slstm.Evaluate_Model(slstm.test_X, slstm.test_y)

if __name__ == '__main__':
    # 对下载的原始文件raw.csv, 开始训练模型
    run()

    # 加载训练好的模型文件,直接进行预测和评价
    #run_loadmodel_predict()
