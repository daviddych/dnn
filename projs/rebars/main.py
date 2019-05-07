#!/usr/bin/env python
#coding:utf-8
from pandas import read_csv
from collections import defaultdict
import cv2
import os
from xml.dom.minidom import Document

def load_train_labels():
    # load rectangles of all training images.
    dataset = read_csv(r'data/train_labels.csv')
    ids = dataset['ID'].get_values()
    rects = dataset['   Detection'].get_values()  # spaces are needed.
    
    img_rects = defaultdict(list)
    for i in range(0, len(ids)):
        img_rects[ids[i]].append(rects[i].split( ))
        
    return img_rects

def main():
    img_rects = load_train_labels()
    
    draw_rects(img_rects)

def draw_rects(img_rects):
    win_title = 'image'
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    for key in img_rects:
        filename = os.path.join('data/train_dataset',key)
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        for (l,t,r,b) in img_rects[key]:
            cv2.rectangle(img, (int(l),int(t)), (int(r),int(b)), (255,0,0), 4)

        cv2.imshow('image', img)
        cv2.resizeWindow('image', 1280, 960)
        if cv2.waitKey(50) == 27:  # & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
            
def writeInfor2Xml(filename, img_rects):
    # 创建dom文档
    doc = Document()

    # 创建根节点
    annotation_list = doc.createElement('annotation verified="no"')

    # 根节点插入dom树
    doc.appendChild(annotation_list)

    # 创建节点<folder>及其文本节点, 并将folder节点插入父节点annotation_list下
    folder_node = doc.createElement('folder')
    folder_text = doc.createTextNode('JPEGImages')
    folder_node.appendChild(folder_text)
    annotation_list.appendChild(folder_node)



if __name__ == '__main__':
    main()