#!/usr/bin/env python
#coding:utf-8

import cv2

''' 1.特征白色 - 黑色'''
''' 2.整个区域*权重-黑色区域*权重'''
''' 3.(p1-p2-p3+p4) * w'''

''' 1.haar魔板, 上下, 左右. image size 1008100 10*10 100次, step=10  '''
''' 模板,滑动,缩放: 11*11  '''
''' 14个模板, 20级缩放, (1080/2)*(720/2)图像划框  '''
''' 积分图: '''
''' haar + adaboost ===>> face识别应用 '''

# 1. 初始化数据权重值分布, 所有权值必须相等
# 2. 遍历阈值 P
# 3. G1 (x)
# 4. 更新权重分布: eg, 0.2 0.2 0.2 0.7
# 5. 训练终止条件判断, for count or 误差p
def run(img_file, face_fier, eye_fier):
    gray = cv2.imread(img_file, 0)

    # 1.3: 模板的比例缩放系数   5: 人脸最下不能小于5个像素
    faces = face_fier.detectMultiScale(gray, 1.3, 5)
    print("face = ", len(faces))
    for (x,y, w, h) in faces:
        cv2.rectangle(gray, (x,y), (x+w, y+h), (255),2)

        # 人眼检测一定是在人脸上
        roi_face = gray[y:y+h, x:x+w]
        eyes = eye_fier.detectMultiScale(roi_face, 1.3, 3)
        if len(eyes) == 2:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_face, (ex,ey), (ex+ew, ey+eh), (255),2)

    cv2.imshow("src gray", gray)
    cv2.waitKey(0)


def load_classifier(face_file, eye_file):
    ''' 健在眼睛和脸分类器 '''
    face_fier = cv2.CascadeClassifier(face_file)
    eye_fier = cv2.CascadeClassifier(eye_file)

    return face_fier, eye_fier


if __name__ == '__main__':
    eye_file = "data/haarcascade_eye.xml"
    face_file = "data/haarcascade_frontalface_default.xml"
    img_file = "data/23.png"

    face_fier, eye_fier = load_classifier(face_file, eye_file)

    run(img_file, face_fier, eye_fier)