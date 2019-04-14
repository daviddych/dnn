#!/usr/bin/env python
#coding:utf-8

import cv2
import numpy as np

# 分块+ 降低图像灰度级

def assert_pixel(x):
    if x > 255:
        x = 255
    elif x <0:
        x = 0

    return x


if __name__ == '__main__':
    img = cv2.imread('data/cat.jpg', 1)
    rows, cols, chs = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dst = np.zeros((rows, cols, chs), np.uint8)

    array1 = np.zeros(8, np.uint8)

    for i in range(4, rows - 4):
        for j in range(4, cols - 4):
            for m in range(-4, 4):
                for n in range(-4, 4):
                    p1 = int(gray[i+m, j +n]/32)
                    array1[p1] = array1[p1] + 1

            currentMax = array1[0]
            l = 0
            for k in range(0, 8):
                if currentMax < array1[k]:
                    currentMax = array1[k]
                    l = k

            for m in range(-4, 4):
                for n in range(-4, 4):
                    if gray[i +m, j +n] >= (l *32) and gray[i+m, j+n] <=((l+1)*32):
                        (b,g,r)=img[i+m, j+n]

            dst[i, j] = (b, g, r)

    cv2.imshow('dst', dst)
    cv2.waitKey(0)


