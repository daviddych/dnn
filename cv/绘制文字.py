#!/usr/bin/env python
#coding: utf-8

import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('data/cat.jpg',1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (100,100),(200,250), (0, 255, 0), 3)
    cv2.putText(img, 'this is a cat', (100,100), font, 2, (200,100,255),2, cv2.LINE_AA)

    cv2.imshow('src', img)
    cv2.waitKey(0)