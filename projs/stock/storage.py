#!/usr/bin/env python
#coding:utf-8


'''
1. 首先确保本地安装MySQL数据库, 然后创建一个stock数据库
2. 运行该脚本从网络上抓取股票数据存入stock数据库中, stock_name存储股票名和代码, 股票数据存储在以股票名命名的数据表中
'''

import datetime
from mysql_stockdb import MySql_StockDB
from spyderstock import SpyderStock

import time
from functools import wraps


# 定义一个装饰器来测量函数的执行时间
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running {}: {} seconds".format(function.__name__, str(t1 - t0)))
        return result

    return function_timer


@fn_timer
def run():
    # 从网上抓取数据
    stockInfo = {'泸州老窖': '000568.SZ', '南京熊猫': '600775.SS', '口子窖': '603589.SS', 'ST新光':'002147.SZ', '荣盛发展':'002146.SZ'}
    webdata = SpyderStock(stockInfo,datetime.datetime(2000, 3, 10))
    # webdata.print() # 显示抓取到的数据

    # 将数据存储到mysql数据库中
    database = MySql_StockDB(database='stock', user='root', passwd='dyc')
    database.insert_stocks(webdata.stocks)
    database.fetch('stock_name')



if __name__ == '__main__':
    run()