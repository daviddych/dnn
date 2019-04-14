#!/usr/bin/env python
#coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 彩色直方图均衡化
def color_acc_histgram(img):
    simg = cv2.split(img)
    for i in range(0, 3):
        simg[i] = gray_acc_histgram(simg[i])


# 灰度直方图均衡化
def gray_acc_histgram(gray):
    # 计算概率分布
    count = np.zeros(256, np.float)
    for i in range(0, gray.shape[0]):
        for j in range(0, gray.shape[1]):
            id = gray[i,j]
            count[id] = count[id] + 1

    sumpixels = gray.shape[0] * gray.shape[1]
    for i in range(0, 256):
        count[i] = count[i] / sumpixels

    # 计算累计概率分布
    sum1 = float(0)
    for i in range(0, 256):
        sum1 = sum1 + count[i]
        count[i] = sum1

    # 计算一个像素映射表
    map1 = np.zeros(256, np.uint8)
    for i in range(0,256):
        map1[i] = np.uint8(count[i] * 255)

    # 映射
    for i in range(0, gray.shape[0]):
        for j in range(0, gray.shape[1]):
            id = gray[i,j]
            gray[i,j] = map1[id]

    cv2.imshow('after', gray)

# 利用累计直方图均衡化增强图像对比度
def hist(img, type=1):
    '''计算累计概率'''
    if type == 1:
        if img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        gray_acc_histgram(gray)
    else:
        color_acc_histgram(img)


    cv2.imshow('before', gray)



# 图像的直方图信息
def gray_histgram(gray, color=0):
    '''统计概率,绘制图像的直方图分布'''
    count = np.zeros(256, np.float)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            id = gray[i,j]
            count[id] = count[id] + 1

    sumpixels = gray.shape[0] * img.shape[1]
    for i in range(0, 256):
        count[i] = count[i] / sumpixels

    x = np.linspace(0, 255, 256)
    clr = ['r','g','b']
    plt.figure(color)
    plt.bar(np.linspace(0, 255, 256), count, 0.9, alpha=1, color=clr[color])
    plt.show()

# 图像直方图信息
def color_histgam(img):
    if img.shape[2] == 3:
        simg = cv2.split(img)
        for i in range(0, 3):
            gray_histgram(simg[i], i)
    elif img.shape[2] == 1:
        gray_acc_histgram(img)



if __name__ == '__main__':
    img = cv2.imread('data/cat.jpg', 1)
    cv2.imshow('src', img)

    color_histgam(img)

    gray_acc_histgram(img)


