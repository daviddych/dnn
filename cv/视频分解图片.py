#!/usr/bin/env python
#coding:utf-8

import cv2
import os
import re
import numpy as np

def video_split_images(vedio_file, imgs_dir):
    cap = cv2.VideoCapture(vedio_file)
    if cap.isOpened() == False:
        print("Failed to open the video file")
        return

    fps = cap.get(cv2.CAP_PROP_APERTURE)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(fps, width, height)

    i = 0
    while(True):
        if i == 10:
            break
        else:
            (flag, frame) = cap.read()
            if flag == True:
                filename = 'image'+str(i) + '.jpg'
                filename = os.path.join(imgs_dir, filename)
                cv2.imwrite(filename, frame,[cv2.IMWRITE_JPEG_QUALITY,100])
                print(filename)


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def images_merage_video(imgs_dir, video_file):
    # -1:选择一个支持的解码器, 5:FP/s, sz:写入对象大小
    sz = (320, 240)
    videowrite = cv2.VideoWriter(video_file, cv2.CAP_PROP_FOURCC, 5, sz)

    # 循环写图像
    for filename in image_files_in_folder(imgs_dir):
        img = cv2.imread(filename)
        img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.resize(gray, sz, gray)
        #cv2.imshow("s", gray)
        videowrite.write(gray)
        #cv2.waitKey(5)
        print(filename)



if __name__ == '__main__':
    #video_split_images("1.mp4")
    images_merage_video('/home/daiyucheng/data/wider_face/WIDER_val/images/0--Parade','/home/daiyucheng/222.avi')
    print("End")