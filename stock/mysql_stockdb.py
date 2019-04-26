#!/usr/bin/env python
#coding: utf-8
'''
利用pymysql操作mysql数据库, 将从网站上获取的股票数据存储在数据库中,
首先要在终端安装pymysql,然后创建stock数据库;随后,运行改程序,会在stock数据库中创两个表格, 最后调用insert_stock()
stock: 数据库
    stock_name: 表一 
        +-------+----------+------+-----+---------+----------------+
        | Field | Type     | Null | Key | Default | Extra          |
        +-------+----------+------+-----+---------+----------------+
        | id    | int(11)  | NO   | PRI | NULL    | auto_increment |
        | name  | char(20) | NO   | UNI | NULL    |                |
        | code  | char(20) | NO   | UNI | NULL    |                |
        +-------+----------+------+-----+---------+----------------+

    stock_dataframe: 表二
        +----------+----------+------+-----+---------+----------------+
        | Field    | Type     | Null | Key | Default | Extra          |
        +----------+----------+------+-----+---------+----------------+
        | id       | int(11)  | NO   | PRI | NULL    | auto_increment |
        | stock_id | int(11)  | NO   | MUL | NULL    |                |
        | Hight    | char(20) | NO   |     | NULL    |                |
        | Low      | char(20) | NO   |     | NULL    |                |
        | Open     | char(20) | NO   |     | NULL    |                |
        | Close    | char(20) | NO   |     | NULL    |                |
        +----------+----------+------+-----+---------+----------------+
'''

import pymysql
import random
import numpy as np
import pandas as pd

class MySql_StockDB(object):
    def __init__(self, host='127.0.0.1', port=3306, database='stock', user='root', passwd='dyc'):
        ''' 打开数据库,并确保两个表格<stock_name, stock_dataframe>存在, 不存在就创建 '''
        #数据库名
        self.database = database

        # 打开数据库
        self.open_database(host, port, database, user, passwd)

        # 获取数据库中的全部表名, 判断stock_name是否存在,如果不存在就创建
        # 创建股票摘要信息表
        self.table_names = self.fetch_table_names()
        self.stock_name = 'stock_name'
        if self.stock_name not in self.table_names:
            sql = "create table " + self.stock_name + '''(
                   id int PRIMARY KEY NOT NULL AUTO_INCREMENT,
                   name CHAR(20) not null unique,
                   code CHAR(20) not null unique
                  )engine = InnoDB AUTO_INCREMENT = 0 default charset = utf8;'''  # 注意编码,否则无法存储汉字

            self.create_table(sql, table_name=self.stock_name)

    def open_database(self, host='127.0.0.1', port=3306, database='stock', user='root', passwd='dyc'):
        ''' 打开数据库,并建立链接 '''
        print("open database {}".format(database))

        # 创建连接, 首先需要在命令行终端: mysql -u root -p;  show databases;  create database stock;  show databases
        self.conn = pymysql.connect(host=host, port=port, user=user, passwd=passwd, db=database, charset='utf8')

        # 创建游标对象
        self.cursor = self.conn.cursor()

    def close_database(self):
        ''' 关闭数据库及与其之间的链接 '''
        print("close database:", self.database)

        # 再次提交，,防止修改内容丢失,不然无法保存新建或者修改的数据
        self.conn.commit()

        # 关闭游标
        self.cursor.close()
        # 关闭连接
        self.conn.close()

    def fetch_table_names(self):
        ''' 获取self.database中的全部表名 '''
        sql = "select table_name from information_schema.tables where table_schema='{}'".format(self.database)

        # 执行SQL，并返回受影响行数
        effect_row = self.cursor.execute(sql)
        if effect_row > 0:
            table_names = []
            for tab_name in self.cursor.fetchall():
                table_names.append(tab_name[0])

            # print('共查询到', self.cursor.rowcount, '条数据: ', table_names)
            return table_names

    def delete_table(self, table_name):
        ''' 删除self.database中的表table_name '''
        # 使用 execute()方法执行SQL, 如果表存在则删除
        sql="drop table IF EXISTS " + table_name
        print("sql = ", sql)

        # 使用 execute()方法执行SQL
        self.cursor.execute(sql)

        # 提交
        self.conn.commit()

    def create_table(self, sql, table_name):
        ''' 在self.database中创建表table_name '''
        self.cursor.execute(sql)
        self.conn.commit()

    def __del__(self):
        ''' 析构函数中调用释放资源,关闭数据库,断开连接 '''
        self.close_database()

    def clear_table(self, tables):
        ''' 清空表中的内容 '''
        for table in tables:
            self.cursor.execute("delete from {}".format(table))
            self.conn.commit()

    def create_stocktable(self, table, stock):
        ''' 创建股票实际数据信息表,每一个股票会创建一个这样的数据表, 表名就是股票名字 '''
        self.table_names = self.fetch_table_names()
        # 创建股票实际数据信息表 PRIMARY KEY (id),
        if table not in self.table_names:
            sql = 'create table ' + table + '''
                    (  
                       datetime CHAR(20) NOT NULL,
                       stock_id int NOT NULL,
                       High double NOT NULL,
                       Low double NOT NULL,
                       Open double NOT NULL,
                       Close double NOT NULL,
                       Volume double NOT NULL,
                       AdjClose double NOT NULL,
                       PRIMARY KEY (datetime),
                       FOREIGN KEY (stock_id) REFERENCES 
                    ''' + self.stock_name + "(id))engine = InnoDB AUTO_INCREMENT = 0 default charset = utf8;"
            print(sql)

            self.create_table(sql, table_name=table)


    def insert_stock(self, table_name, stock_id, stock):
        # 如果需要就创建表格
        self.create_stocktable(table_name, stock)
        # 向表格中插入数据, 注意无论mysql数据库中表实际是int,float还是string, 此处一律使用%s
        sql_dataframe = "INSERT IGNORE INTO {} (datetime, stock_id, High, Low, Open, Close, Volume, AdjClose ) " \
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)".format(table_name)

        date_ = stock.index
        vals = stock.values  # 这种方式可以一次性获取全部的数据 np.array 二维数组.
        # highs = stock['High'].get_values()
        # lows = stock['Low'].get_values()
        # opens = stock['Open'].get_values()
        # closes = stock['Close'].get_values()
        # volume = stock['Volume'].get_values()
        # AdjClose = stock['Adj Close'].get_values()
        # print("长度 = {} -{}: {}".format(date_.shape[0], highs.shape[0], vals.shape))
        for j in range(0, date_.shape[0]):
            try:
                # self.cursor.execute(sql_dataframe, (
                # str(date_[j]), str(stock_id), str(highs[j]), str(lows[j]), str(opens[j]), str(closes[j]),
                # str(volume[j]), str(AdjClose[j])))

                self.cursor.execute(sql_dataframe, (
                    str(date_[j]), str(stock_id), str(vals[j,0]), str(vals[j,1]), str(vals[j,2]), str(vals[j,3]),
                    str(vals[j,4]), str(vals[j,5])))

            except:
                pass

        self.conn.commit()


    def insert_stocks(self, stocks):
        ''' 将多只股票数据插入数据库 '''
        # self.clear_table(['泸州老窖', '{}'.format(self.stock_name)]) # 清空表
        sql = "select * from " + self.stock_name
        rows = self.cursor.execute(sql)  # 执行sql

        sql_name = "INSERT IGNORE INTO {} (id, name, code) VALUES (%s, %s, %s)".format(self.stock_name)
        for name_, (code_, stock) in stocks.items():
            try:
                rows = self.cursor.execute("select max(id) from stock_name") # 每次从表中查询当前最大id值
                if rows > 0:
                    res = self.cursor.fetchone()
                    if res[0] == None:
                        self.cursor.execute(sql_name, (str(1), name_, code_))  # 空表,插入第一个数据,id=1
                    else:
                        self.cursor.execute(sql_name, (str(res[0]+1), name_, code_))  # 列表格式数据
            except:
                pass

            self.conn.commit()

            # 获取当前股票的主键麻
            try:
                sql = "select id from {} where name='{}'".format(self.stock_name, name_)
                self.cursor.execute(sql)
                stock_id = self.cursor.fetchone()
                self.conn.commit()
            except:
                print("获取主键失败")

            # 将数据插入以股票名为表名的数据表中
            self.insert_stock(name_, stock_id[0], stock)

    def fetch(self, table_name):
        ''' 查询表中的全部内容 '''
        sql = "select * from " + table_name
        rows = self.cursor.execute(sql)  # 执行sql

        if table_name is 'stock_name':
            # 查询所有数据，返回结果默认以元组形式，所以可以进行迭代处理
            print("\n{}:".format(table_name))
            dict1 = {'name': [], 'code': []}
            for i, v in enumerate(self.cursor.fetchall()):
                print(v)
                dict1['name'].append(v[1])
                dict1['code'].append(v[2])

            dataset = pd.DataFrame(dict1)
        else:
            print("\n{}:".format(table_name))
            #dict1 = {'High': [], 'Low': [], 'Open': [], 'Close': [], 'Volume': [], 'Adj Close': []}
            vals = np.zeros((rows, 6))
            date_ = np.zeros((rows,1), dtype='datetime64[D]')
            for i, v in enumerate(self.cursor.fetchall()):
                date_[i] = v[0]
                vals[i,:] = v[2:]

            dataset = pd.DataFrame(vals, index=date_, columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])

        return dataset


def run():
    db = MySql_StockDB(database='stock', user='root', passwd='dyc')
    tables = db.fetch_table_names()
    print(tables)

    db.fetch('stock_name')
    res = db.fetch('泸州老窖')

    del db

if __name__ == '__main__':
    run()

    #  = stock[''].get_values()
    #  = stock[''].get_values()
    #  = stock[''].get_values()
    #  = stock[''].get_values()
    #  = stock[''].get_values()
    #  = stock[''].get_values()
