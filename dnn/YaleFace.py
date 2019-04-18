#!/usr/bin/env python
#coding:utf-8
import tensorflow as tf
import numpy as np
import scipy.io as sio
import pymysql

def load_data(filename = 'Yale_64x64.mat'):
    f = open(filename, 'rb')
    mdict = sio.loadmat(f)

    train_data = dict['fea']
    train_label = dict['gnd']

    # 函数shuffle与permutation都是对原来的数组进行重新洗牌（即随机打乱原来的元素顺序）；
    # 区别在于shuffle直接在原来的数组上进行操作，改变原来数组的顺序，无返回值。
    # 而permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
    train_data = np.random.permutation(train_data)
    train_label = np.random.permutation(train_label)
    test_data = train_data[0:64]
    test_label = train_label[0:64]
    np.random.seed(100)
    test_data = np.random.permutation(test_data)
    test_label = np.random.permutation(test_label)

    train_data = train_data.reshape(train_data.shape[0], 64, 64, 1).astype(np.float32)
    train_label_new = np.zeros((165, 15)) # 165image --- 15person.
    for i in range(0, 165):
        j = int(train_label[i,0]) - 1
        train_label_new[i, j] = 1

    test_data = test_data.reshape(test_data.reshape())

def conn_mysql():
    # 创建连接
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='dyc', db='mysql', charset='utf8')
    # 创建游标对象
    cursor = conn.cursor()

    # 使用 execute()方法执行SQL, 如果表存在则删除
    cursor.execute("drop table IF EXISTS employee_dyc")

    # 使用预处理语句创建表
    # sql = '''create table employee_dyc(
    #        id int PRIMARY KEY NOT NULL auto_increment,
    #        name CHAR(20) not null,
    #        passwd CHAR(20),
    #        age int,
    #        sex char(1)
    #       )'''
    #
    # cursor.execute(sql)
    #
    # # 使用
    # # 使用 execute()方法执行SQL, 并返回收影响行数
    # effect_row = cursor.execute("select * from user")
    # print(effect_row)

    # 执行SQL，并返回受影响行数
    # effect_row = cursor.execute("update tb7 set pass = '123' where nid = %s", (11,))

    # 执行SQL，并返回受影响行数,执行多次
    # effect_row = cursor.executemany("insert into tb7(user,pass,licnese)values(%s,%s,%s)", [("u1","u1pass","11111"),("u2","u2pass","22222")])

    # 提交，不然无法保存新建或者修改的数据
    conn.commit()

    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()

def run():
    conn_mysql()
    pass

if __name__ == '__main__':

    run()
    print("End")