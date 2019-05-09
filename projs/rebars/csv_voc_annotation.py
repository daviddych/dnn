#!/usr/bin/env python
#coding:utf-8

from csv_label import covert_csv_to_xml
import os

cls_id = 0

def to_voc_annotation(img_rects, annotation_filename):

    annotation_file = open(annotation_filename, 'w', encoding='UTF-8')

    for imgname in img_rects:
        annotation_file.write(os.path.join(os.getcwd(), r'data/train_dataset/{}.jpg'.format(imgname)))

        for (l, t, r, b) in img_rects[imgname]:
            annotation_file.write(" " + ",".join([str(a) for a in (l, t, r, b)]) + ',' + str(cls_id))

        annotation_file.write('\n')
    annotation_file.close()

if __name__ == '__main__':
    img_rects = covert_csv_to_xml(csvfile=r'data/train_labels.csv', img_folder=r'data/train_dataset',
                      xmlfolder=r'data/train_Annotations/')

    to_voc_annotation(img_rects, 'train.txt')