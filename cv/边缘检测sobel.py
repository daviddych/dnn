#!/usr/bin/env python
#coding:utf-8

import cv2
import numpy as np
import math
from skimage import data,img_as_float

# sobel
# [1  2  1               [1  0  -1
#  0  0  0                2  0  -2
# -1 -2 -1]               1  0  -1]

img = cv2.imread('data/cat.jpg', 1)
rows, cols, chs = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_float = img_as_float(gray/255.0)

dst = np.zeros((rows, cols, 1), np.uint8)
for i in range(1, rows - 3):
    for j in range(1, cols - 3):
        gy = gray_float[i, j] + gray_float[i - 1, j - 1] + 2 * gray_float[i, j - 1] + gray_float[i + 1, j - 1] - gray_float[i - 1, j + 1] - 2 * gray_float[i, j + 1] - gray_float[i + 1, j + 1]
        gx = gray_float[i, j] + gray_float[i - 1, j - 1] + 2 * gray_float[i - 1, j] + gray_float[i - 1, j + 1] - gray_float[i + 1, j + 1] - 2 * gray_float[i + 1, j] - gray_float[i + 1, j + 1]
        grad = np.uint8(math.sqrt(gx * gx + gy * gy) * 255.0)

        if grad > 50:
            dst[i, j] = 255
        else:
            dst[i,j] = 0

cv2.imshow('dst', dst)
cv2.waitKey(0)