#!/usr/bin/env python
#coding:utf-8
''' 从雅虎财经网站抓取股票数据 '''

import datetime
import pandas_datareader.data as web
from liveplot import LivePlot
# import liveplot
import mysql_stockdb


class SpyderStock(object):
    def __init__(self, name_codes):
        ''' 从雅虎财经网站下载指定的股票(名字+股票代码)数据 '''
        self.stocks = {}
        self.drawer = LivePlot()

        # 遍历下载股票数据
        for name, code in name_codes.items():
            stock = self.download_stock(code)
            if len(stock) == 0:
                print("Failed to spyder ", name, " data!")
            else:
                # 计算变换并添加进去
                self.stocks[name] = (code, stock)#self.addColumn_change(stock)

    def download_stock(self, code, start_date=datetime.datetime(2019, 4, 1), end_date=datetime.date.today()):
        ''' 从雅虎财经上下载特定时间段内的特定代码的股票数据 '''
        stock = web.DataReader(code, 'yahoo', start_date, end_date)
        if len(stock) == 0:
            print("DataReader failed")
            return None
        return stock

    def addColumn_change(self, stock):
        ''' DataFrame数据中增加涨/跌幅列，涨/跌=（当日Close-上一日Close）/上一日Close*100% '''
        change = stock.Close.diff()

        # 第一行不再有数据, 所以计算的值为NAN, 对缺失的数据用涨跌值的均值就地替代NaN。
        change.fillna(change.mean(), inplace=True)

        stock["Change"] = change

        # 计算涨跌幅度有两种方法，pct_change()算法的思想即是第二项开始向前做减法后再除以第一项，计算得到涨跌幅序列。
        stock['pct_change'] = (stock['Change'] / stock['Close'].shift(1))  #
        stock['pct_change1'] = stock.Close.pct_change()

        return stock

    def print(self):
        if len(self.stocks) > 0:

            for name, (code, stock) in self.stocks.items():
                self.drawer.open_close_bar(stock)
                print("name = ", name)

def datatime2str(stock):
    date_ = stock.index
    for i in range(0, date_.shape[0]):
        dt=date_[i].strftime("...%Y-%m-%d %H:%M:%S")
        print(date_[i], dt)
        
def run():
    #stockInfo = {'泸州老窖': '000568.SZ', '南京熊猫': '600775.SS', '口子窖': '603589.SS', '中兴通信': '000063.SZ',
    #             "*ST凯迪": '000939.SZ', '大恒科技': '600288.SS'}
    stockInfo = {'泸州老窖': '000568.SZ'}
    dd = SpyderStock(stockInfo)
    dd.print()

    # return

    db = mysql_stockdb.MySql_StockDB(database='stock', user='root', passwd='dyc')
    db.insert_stocks(dd.stocks)
    db.fetch('stock_name')
    db.fetch('stock_dataframe')

if __name__ == '__main__':
    run()