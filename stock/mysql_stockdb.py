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

class MySql_StockDB(object):
    def __init__(self, host='127.0.0.1', port=3306, database='stock', user='root', passwd='dyc'):
        ''' 打开数据库,并确保两个表格<stock_name, stock_dataframe>存在, 不存在就创建 '''
        #数据库名
        self.database = database

        # 打开数据库
        self.open_database(host, port, database, user, passwd)

        # 获取数据库中的全部表名, 判断stock_name是否存在,如果不存在就创建
        # 创建股票摘要信息表
        self.table_names = self.fetch_tabel_names()
        stock_name = 'stock_name'
        if stock_name not in self.table_names:
            sql = "create table " + stock_name + '''(
                   id int PRIMARY KEY NOT NULL,
                   name CHAR(20) not null unique,
                   code CHAR(20) not null unique
                  )engine=InnoDB default charset=utf8'''  # 注意编码,否则无法存储汉字

            self.create_table(sql, table_name=stock_name)

        # 创建股票实际数据信息表 PRIMARY KEY (id),
        table_dataframe = 'stock_dataframe'
        if table_dataframe not in self.table_names:
            sql = 'create table ' + table_dataframe + '''
            (  
               id int NOT NULL,
               stock_id int NOT NULL,
               High CHAR(20) NOT NULL,
               Low CHAR(20) NOT NULL,
               Open CHAR(20) NOT NULL,
               Close CHAR(20) NOT NULL,
               FOREIGN KEY (stock_id) REFERENCES 
            ''' + stock_name + "(id))"
            print(sql)

            self.create_table(sql, table_name=table_dataframe)

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

    def fetch_tabel_names(self):
        ''' 获取self.database中的全部表名 '''
        sql = "select table_name from information_schema.tables where table_schema='{}'".format(self.database)
        print("fetch_table_names = {}".format(sql))

        # 执行SQL，并返回受影响行数
        effect_row = self.cursor.execute(sql)
        table_names = []
        for tab_name in self.cursor.fetchall():
            table_names.append(tab_name[0])

        print('共查询到', self.cursor.rowcount, '条数据: ', table_names)
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

    def clear_table(self, table_names):
        ''' 清空表中的内容 '''
        for table_name in table_names:
            self.cursor.execute("delete from {}".format(table_name))
            self.conn.commit()

    def insert_stocks(self, stocks):

        # 清空表
        self.clear_table(['stock_dataframe', 'stock_name'])

        sql_name = "INSERT INTO stock_name (id, name, code) VALUES (%s, %s, %s)"
        sql_dataframe = "INSERT INTO stock_dataframe (id, stock_id, High, Low, Open, Close) VALUES (%s, %s, %s, %s, %s, %s)"
        for i, (name_, (code_, stock)) in enumerate(stocks.items()):
            try:
                self.cursor.execute(sql_name, (str(i), name_, code_))  # 列表格式数据
                print("Sucess instert")
                self.conn.commit()
            except:
                print("重复了无需重复插入1111............")

            sql = "select id from stock_name where name='{}'".format(name_)
            try:
                rows = self.cursor.execute(sql)
                print('受影响的行=', rows)
            except:
                print("重复了插入失败2222............")

            stock_id = self.cursor.fetchone()
            print("stock_id = ", stock_id[0])
            self.conn.commit()

            highs = stock['High'].get_values()
            lows = stock['Low'].get_values()
            opens = stock['Open'].get_values()
            closes = stock['Close'].get_values()
            for j in range(0, highs.shape[0]):
                # 列表格式数据
                try:
                    self.cursor.execute(sql_dataframe, (str(j), str(stock_id[0]), str(highs[j]), str(lows[j]), str(opens[j]), str(closes[j])))
                    self.conn.commit()
                except:
                    print("重复无需重复插入3333............")

    def instert(self, table_name):
        sql = "INSERT INTO stock_name (name, code) VALUES (%s, %s)"
        self.cursor.execute(sql, ('Wilson'+str(random.randint(0,10000)), 'Champs-Elysees'+str(random.randint(0,10000))))  # 列表格式数据
        print('sql = ', sql)
        self.conn.commit()

    def fetch(self, table_name):
        ''' 查询表中的全部内容 '''
        print("\n查询表:" + table_name)
        # 查询数据
        sql = "select * from " + table_name
        print('sql = ', sql)
        self.cursor.execute(sql)  # 执行sql

        # 查询所有数据，返回结果默认以元组形式，所以可以进行迭代处理
        for v in self.cursor.fetchall():
            print(v)
        print('共查询到：', self.cursor.rowcount, '条数据。')


def run():
    db = MySql_StockDB(database='stock', user='root', passwd='dyc')
    tables = db.fetch_tabel_names()

    db.instert('stock_name')
    db.fetch('stock_name')


    del db

if __name__ == '__main__':
    run()