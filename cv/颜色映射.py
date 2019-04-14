#!/usr/bin/env python
#coding:utf-8

import cv2
import numpy as np

# (r,g,b) --> (r, b * 1.5, g * 1.3)

def assert_pixel(x):
    if x > 255:
        x = 255
    elif x <0:
        x = 0

    return x


if __name__ == '__main__':
    img = cv2.imread('data/cat.jpg', 1)
    rows, cols, chs = img.shape

    dst = np.zeros((rows, cols, chs), np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            (b, g, r) = img[i, j]

            dst[i, j] = (b, assert_pixel(g * 1.5), assert_pixel(r * 1.3))

    cv2.imshow('dst', dst)
    cv2.waitKey(0)


