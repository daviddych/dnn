#!/usr/bin/env python
#coding:utf-8

''' 运行该脚本预测股票开盘价之前,首先运行storage.py脚本, 从网上抓取数据存入本地MySQL数据库'''

from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
import tensorflow
import os

from mysql_stockdb import MySql_StockDB

class StockLSTM(object):
    ''' 对股票时序数据利用LSTM模型训练和预测 '''
    def __init__(self, stock):
        if type(stock) is not DataFrame:
            print("stock数据初始化错误")
            return

        self.stock = stock # 股票数据
        self.tbCallBack = tensorflow.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

        self.save_model_img = 'image/stockSeries_lstm.png'
        self.save_model_file = 'image/stockSeries_lstm.h5'
        self.n_hours = 3
        self.n_features = self.stock.values.shape[1]

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

    def scale_fit(self, values):
        ''' 将新的预测数据和全部训练数据一起, 获得scaler '''
        # normalize features, sklearn中的这个归一化是对列进行归一化, 公式如下
        #                       X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        #                       X_scaled = X_std * (max - min) + min
        # ensure all data is float
        values = values.astype('float32')
        self.scaler_cols = values.shape[1]
        self.scaler = MinMaxScaler(feature_range=(0,1))
        scaled = self.scaler.fit_transform(values)
        return scaled

    def scale_invfit(self, values):
        ''' 将values数据变换会原始数据尺度空间, 要求两次调用scaler.fit_transform() 和 scaler.inverse_transform()时 传入传参数列相同 '''
        if values.shape[1] != self.scaler_cols:
            print("请确保values的列数为{}".format(self.scaler_cols))
            return None

        return self.scaler.inverse_transform(values)


    def normal_toSupervised(self):
        ''' 对股票数据归一化 + 转成适合监督学习的数据组织形式'''
        values = self.stock.values
        scaled = self.scale_fit(values)
        reframed = self.series_to_supervised(scaled, self.n_hours, 1)
        print("reframed.shape = {}".format(reframed.shape))

        return reframed

    def split_train_test(self, reframed, train_vs_test=0.8):
        ''' 将数据划分为训练集和测试集, train_vs_test表示训练集占整个数据集部分的比例. '''
        values = reframed.values
        n_train_hours = int(values.shape[0] * train_vs_test)
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]
        # split into input and outputs
        n_obs = self.n_hours * self.n_features
        train_x, train_y = train[:, :n_obs], train[:, -self.n_features]
        test_x, test_y = test[:, :n_obs], test[:, -self.n_features]
        # reshape input to be 3D [samples, timesteps, features]
        train_x = train_x.reshape((train_x.shape[0], self.n_hours, self.n_features))
        test_x = test_x.reshape((test_x.shape[0], self.n_hours, self.n_features))
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        return train_x, train_y, test_x, test_y

    def preprocess_data(self, train_vs_test=0.8):
        """ 股票数据预处理, 使其适合深度学习 """
        reframed = self.normal_toSupervised()

        # split into train and test sets
        self.train_X, self.train_y, self.test_X, self.test_y = self.split_train_test(reframed, train_vs_test)
        return self.train_X, self.train_y, self.test_X, self.test_y

    def design(self):
        ''' 设计lstm网络 '''
        self.model = Sequential()
        
        self.model.add(LSTM(50, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
#        self.model.add(LSTM(50, input_shape=(self.train_X.shape[1], self.train_X.shape[2]), return_sequences=True))
#        self.model.add(keras.layers.Dropout(0.4))
#        self.model.add(LSTM(30))
#        self.model.add(keras.layers.Dropout(0.25))
#        keras.layers.BatchNormalization

        self.model.add(Dense(1))
        self.model.compile(loss='mae', optimizer='adam', metrics=[
            'accuracy'])  # 添加: metrics=['accuracy'] 在fit模型返回值中就会有history['acc'] 和 history['val_acc']

        self.model_info(to_file=self.save_model_img)

        return self.model

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

    def train(self, epochs=200, batch_size=32):
        # fit network
        hist = self.model.fit(self.train_X, self.train_y, epochs=epochs, batch_size=batch_size, validation_data=(self.test_X, self.test_y), verbose=2,
                              shuffle=False, callbacks=[self.tbCallBack])  # self.plot_losses  , self.history

        score = self.model.evaluate(self.test_X, self.test_y, verbose=0)
        print("Test score: {}", score[0])
        print("Test accuracy: {}", score[1])

        self.draw_history(hist)
        return self.model

    def model_info(self, to_file=None):
        ''' 打印网络结构, 保存为图像文件 '''
        print(self.model.summary())
        if to_file == None:
            to_file = self.save_model_img

        keras.utils.plot_model(self.model, to_file)

    # 保存训练好的模型
    def save_model(self, filename=None):
        if filename == None:
            filename = self.save_model_file  # self.str_time() + '.h5'

        self.model.save(filename)

    # 加载已经训练好的模型
    def load_model(self, filename=None):
        if filename == None:
            filename = self.save_model_file

        print("模型文件:", filename)
        if os.path.exists(filename) == False:
            print("Cannot find: ", filename)
            exit()

        self.model = keras.models.load_model(filename)
        self.model_info()

    def inv_reframed_y(self, yhat, reframed_X):
        ''' 将预测结果y, 返回原归一化缩放空间 '''
        if (len(reframed_X.shape) == 3):
            reframed_X = reframed_X.reshape((reframed_X.shape[0], self.n_hours * self.n_features))


        # make a prediction: reconstruct the rows with n_features columns suitable for reversing the scaling operation
        predict_data = concatenate((yhat, reframed_X[:, -(self.n_features-1):]), axis=1) # 将reframed_y 和 reframed_x的最后一行重组,新的数组
        # 重组后的数据满足scale_invfit调用要求, 反算回原尺度空间
        data = self.scale_invfit(predict_data)
        # 把临时帮忙的后几列辅助数据剔除
        data = data[:,0]

        return  data

    def Evaluate_Model(self, test_X=None, test_y=None):
        ''' Evaluate the model '''
        if test_X == None or test_y == None:
            test_X = self.test_X
            test_y = self.test_y

        # make a prediction: reconstruct the rows with n_features columns suitable for
        # reversing the scaling operation to get the y and yhat back into the original scale
        # so that we can calculate the RMSE.
        yhat = self.model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], self.n_hours*self.n_features))
        # invert scaling for forecast
        inv_yhat = self.inv_reframed_y(yhat, test_X)
        # The gist of the change is that we concatenate the y or yhat column with the last 7 features of the test
        # dataset in order to inverse the scaling

        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = self.inv_reframed_y(test_y, test_X)

        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y[1:], inv_yhat[:-1]))
        print('Test RMSE: %.3f' % rmse)


    # 根据模型获取预测结果，为了节约计算内存，也是分组（batch）加载到内存
    def predict(self, x, batch_size=128, verbose=0):
        result = self.model.predict(x, batch_size=batch_size, verbose=verbose)


        return result

def run():
    # 1. 从SQL中获取指定股票的时序数据
    db = MySql_StockDB(database='stock', user='root', passwd='dyc')
    tables = db.fetch_table_names()
    print(tables)

    db.fetch('stock_name')
    stock = db.fetch('泸州老窖')

    # 2. 进行LSTM模型训练
    lstm = StockLSTM(stock)
    # 股票数据预处理(归一化缩放,时序转监督), 使其适合深度学习
    train_x, train_y, test_x, test_y = lstm.preprocess_data(0.9)
    print("train_X.shape = ", train_x.shape)
    # lstm.design()
    # lstm.train(10000, 128)
    # lstm.Evaluate_Model()
    # lstm.save_model()
    lstm.load_model()
    print('load model sucess')

    # 3. 进行模型回归预测
    result = lstm.predict(test_x[-4:-1, :])
    print(result.shape)
    print('test_X:', test_x[-4:-1, :])
    print('test_X.shape:', test_x.shape)
    #result = result.reshape((result.shape[0],))
    print("result: {} ---> {}".format(result, lstm.inv_reframed_y(result, test_x[-4:-1, :])))
    
    y_true = test_y[-3:]
    y_true = y_true.reshape((3, 1))
    print("x[-1,:]:{} ---> {}".format(test_y[-3:], lstm.inv_reframed_y(y_true, test_x[-3:])))
    lstm.Evaluate_Model()
    del db


if __name__ == '__main__':
    run()