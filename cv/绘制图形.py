import cv2
import numpy as np

if __name__ == '__main__':
    imgshape = (500, 500, 3)
    dst = np.zeros(imgshape, np.uint8)

    # draw line
    cv2.line(dst, (100,100), (400,400), (0,255,255), 2)
    cv2.line(dst, (100, 200), (400, 200), (0, 255, 255), 10)
    cv2.line(dst, (100, 300), (300, 200), (0, 255, 255), 10, cv2.LINE_AA)

    # draw rectangle
    cv2.rectangle(dst, (50,50), (200,300), (255,0,0), 1) # -1表示填充, 正数表示不填充

    # 绘制圆
    cv2.circle(dst, (250,250), (50), (0,255,0), 2)

    # 绘制椭圆
    cv2.ellipse(dst, (256,256), (150,100), 0, 0, 180, (255,255,0), -1)

    # 绘制多边形
    points = np.array([[150,50],[140,140],[200, 170],[250,250],[150,50]], np.int32)
    points = points.reshape(-1,1,2) # 转置
    cv2.polylines(dst, [points], True, (0,255,255))

    cv2.imshow('dst', dst)
    cv2.waitKey(0)