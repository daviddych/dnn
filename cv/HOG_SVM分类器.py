#!/usr/bin/env python
#coding:utf-8

''' SVM本质是分类器, 寻求一个最优的超平面 '''
''' SVM核: line'''

import cv2
import numpy as np


if __name__ == '__main__':
    # 参数定义
    PosNum = 820
    NegNum = 1931
    winSize = (64, 128)
    blockSize = (16, 16)  # 105 bocks
    blockStride = (8, 8)  # 4 cells
    cellSize = (8, 8)
    nBin = 9

    # 1. hog创建
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBin)

    # 2. SVM创建
    svm = cv2.ml.SVM_create()

    # 3. 计算hog
    featureNum = int(((128-16)/8+1) *((64-16)/8+1)*4*9) # 3780
    featureArray = np.zeros((PosNum+NegNum, featureNum), np.float32)
    labelArray = np.zeros((PosNum + NegNum, 1), np.int32)

    # 4. SVM 监督学习,样本, 标签, svm-> image hog
    for i in range(0, PosNum):
        fileName = 'pos/'+ str(i + 1) + '.jpg'
        img = cv2.imread(fileName)
        hist = hog.compute(img, winStride=(8,8))
        for j in range(0, featureNum):
            featureArray[i,j] = hist[j]
        labelArray[i, 0] = 1

    for i in range(0, NegNum):
        fileName = 'neg/'+ str(i + 1) + '.jpg'
        img = cv2.imread(fileName)
        hist = hog.compute(img, winStride=(8,8))
        for j in range(0, featureNum):
            featureArray[i+PosNum,j] = hist[j]
        labelArray[i, 0] = -1

    # 5.svm属性设置
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(0.01)

    # 7.svm训练
    ret = svm.train(featureArray, cv2.ml.ROW_SAMPLE, labelArray)

    # 8.检测
    alpha = np.zeros((1), np.float32)
    rho = svm.getDecisionFunction(0, alpha)
    print('rho = ', rho, '; alhpa = ', alpha)
    alphaArray = np.zeros((1,1), np.float32)
    supportVarray = np.zeros((1, featureNum), np.float32)
    resultArray = np.zeros((1, featureNum), np.float32)
    alphaArray[0, 0] = alpha
    resultArray = -1 * alphaArray * supportVarray

    # 9.detect
    myDetect = np.zeros(3781, np.float32)
    for i in range(0, 3781):
        myDetect[i] = resultArray[0, i]
    myDetect[3781] = rho[0]

    # 10.构建hog, 完成预测
    myHog = cv2.HOGDescriptor()
    myHog.setSVMDetector(myDetect)
    imageSrc = cv2.imread("test2.jpg")
    objs = myHog.detectMultiScale(imageSrc, 0, (8,8), (32, 32), 1.05, 2)
    x = int(objs[0][0][0])
    y = int(objs[0][0][1])
    w = int(objs[0][0][2])
    h = int(objs[0][0][3])

    cv2.rectangle(imageSrc, (x, y), (x+w, y+h), (255, 255,0), 2)
    cv2.imshow('dst', imageSrc)
    cv2.waitKey(0)