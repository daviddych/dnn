#!/usr/bin/env python
#coding:utf-8

''' SVM本质是分类器, 寻求一个最优的超平面 '''
''' SVM核: line'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


def LinearSVM_HW():
    '''利用线性SVM进行身高体重预测性别 '''
    boy = np.array([[155, 48], [159, 50], [164, 53], [168, 56], [172, 60]])
    girl = np.array([[152, 53], [156, 55], [160, 56], [172, 64], [176, 65]])

    # 0:负样本, 1:正样本
    label = np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])

    data = np.vstack((boy, girl))
    data = np.array(data, dtype=np.float32)

    # 创建SVM分类器,并设置属性
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(0.01)

    # 训练
    result = svm.train(data, cv2.ml.ROW_SAMPLE, label)

    # 预测
    p_data = np.vstack([[167,55], [162,57]]) # 0女生 1男生
    p_data = np.array(p_data, dtype=np.float32)
    print(p_data)
    (r1, r2) = svm.predict(p_data)
    print(r2)

# HOG特征:  1.模块划分, 2.梯度,方向,模板 3.bin投影 4.每个模板hog
# 1.模块划分: image > win > block > cell   64*128  16*16(8*8)  8*8(8*8)
#   在每个win中: bolcks = ((64-16)/8+1) * (128-16/8+1) = 105
#              cells  = 16/8 * 16/8 = 4
#   HoG特征向量维度: 105*6 * 9 = 3780

# 2.每个像素梯度(大小和方向): [1,0,-1] [[1],[0],[-1]] :
#   相邻像素之差: a = p1 * 1 + p2 * 0 + p3 * -1
#   上下像素之差: b
#   f = sqrt(a^2 + b^2)
#   angle = arctan(b/a)

# 3.bin: 每个像素梯度,360方向,划分为9份, bin1:[0,20] [180,200]

# 4. 整个HOG, cell复用

if __name__ == '__main__':
    LinearSVM_HW()