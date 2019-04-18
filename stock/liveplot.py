#!/usr/bin/env python
#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

''' 绘制股票数据图 '''
class LivePlot(object):
    def __init__(self):
        pass

    def open_close_bar(self, stock):
        ''' 绘制每天开闭市的柱状图 '''
        priceOne = np.zeros(2)

        date_ = stock.index
        open_values = stock['Open'].get_values()
        close_values = stock['Close'].get_values()
        for i in range(0, date_.shape[0]):
            priceOne[0] = open_values[i]
            priceOne[1] = close_values[i]

            if priceOne[0] > priceOne[1]:
                plt.plot((date_.date[i], date_.date[i]), priceOne, 'g', lw=3)
            else:
                plt.plot((date_.date[i], date_.date[i]), priceOne, 'r', lw=3)

        plt.xticks(rotation=45)
        plt.xlabel('Date')
        plt.xlabel('Price')
        plt.show()