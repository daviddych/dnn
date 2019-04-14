#!/usr/bin/env python
#coding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

def ImgHist(img, type):
    color = (255, 255, 255)
    wintitle = "gray"
    if type == 31:
        color = (255, 0, 0)
        wintitle = 'B hist'
    elif type == 32:
        color = (0,255,0)
        wintitle = 'G hist'
    elif type == 33:
        color = (0, 0, 255)
        wintitle = 'R hist'

    hist = cv2.calcHist([img],[0],None, [256], [0.0, 255.0])
    minV, maxV, minI, maxI = cv2.minMaxLoc(hist)

    # 创建画布
    histImg = np.zeros([256,256,3], np.uint8)

    for h in range(0, 256):
        tmpNorm = int(hist[h] * 256 / maxV)
        cv2.line(histImg, (h, 256), (h, 256-tmpNorm), color)

    cv2.imshow('histImg', histImg)
    cv2.waitKey(1000)


def gray_hist(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    cv2.imshow('gray_hist', dst)

def color_hist(img):
    simg = cv2.split(img)
    for i in range(0, 3):
        simg[i] = cv2.equalizeHist(simg[i])

    dst = cv2.merge(simg)
    cv2.imshow("color_hist", dst)


def YUV_hist(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    simg = cv2.split(yuv)
    for i in range(0, 3):
        simg[i] = cv2.equalizeHist(simg[i])

    dst = cv2.merge(simg)

    dst = cv2.cvtColor(dst, cv2.COLOR_YUV2BGR)
    cv2.imshow("yuv_hist", dst)

if __name__ == '__main__':
    img = cv2.imread('data/cat.jpg', 1)
    cv2.imshow('src', img)

    ch_imgs = cv2.split(img)
    for i in range(0,3):
        ImgHist(ch_imgs[i], 31 + i)

    gray_hist(img)

    color_hist(img)

    YUV_hist(img)

    cv2.waitKey(0)
