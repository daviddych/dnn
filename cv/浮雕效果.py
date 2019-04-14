#!/usr/bin/env python
#coding:utf-8

import cv2
import numpy as np

# 相邻像素做差

img = cv2.imread('data/cat.jpg', 1)
rows, cols, chs = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = np.zeros((rows, cols, 1), np.uint8)
for i in range(0, rows):
    for j in range(0, cols - 1):
        gray0 = int(gray[i,j])
        grayP1 = int(gray[i, j+ 1])
        newP = gray0 - grayP1 + 150
        if newP > 255:
            newP =255
        elif newP < 0:
            newP = 0

        dst[i,j] = newP

cv2.imshow('dst', dst)
cv2.waitKey(0)