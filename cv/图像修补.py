#!/usr/bin/env python
#coding:utf-8
import cv2
import numpy as np

# 在图像中绘制一个十字叉
def damage(img):
    for i in range(200, 250):
        for j in range(-1, 1):
            img[i, 200 + j] = (255,255,255)
            img[225 + j, i-25] = (255, 255, 255)

    cv2.imshow('damage_image', img)

    return img

def repair(img):
    ''' 图片修补 '''

    # 定义一个图片数组
    paint = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    # 指定mask
    for i in range(200, 250):
        for j in range(-1, 1):
            paint[i, 200 + j] = 255
            paint[225 + j, i-25] = 255
    cv2.imshow('paint', paint)

    # 图片修补
    rimg = cv2.inpaint(img, paint, 3, cv2.INPAINT_TELEA)
    cv2.imshow("reparimage", rimg)
    return  rimg

if __name__ == '__main__':
    img = cv2.imread('data/cat.jpg', 1)
    cv2.imshow('src', img)

    dimg = damage(img)
    rimg = repair(dimg)

    cv2.waitKey(0)
