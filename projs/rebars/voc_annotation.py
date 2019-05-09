import xml.etree.ElementTree as ET
from os import getcwd
import os

# 配置文件中每一行表示一个类标签名
def get_classes(classes_path):
    with open(classes_path, encoding='UTF-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

# 从配置文件中解析类标签名
classes = get_classes('model/voc_classes.txt')

# 依据图片名,从对应的xml文件中获得矩形框,并存储到list_file文件中
def convert_annotation(img_name, annotation_file):
    in_file = open('data/train_Annotations/{}.xml'.format(img_name), encoding='UTF-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        annotation_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for image_set in ['train', 'trainval', 'val', 'test']:
    image_ids = open('data/splitset/{}.txt'.format(image_set), encoding='UTF-8').read().strip().split()
    filename = '{}.txt'.format(image_set)
    print(filename)
    annotation_file = open(filename, 'w', encoding='UTF-8')
    for image_id in image_ids:
        annotation_file.write(os.path.join(os.getcwd(), r'data/train_dataset/{}.jpg'.format(image_id)))
        convert_annotation(image_id, annotation_file)
        annotation_file.write('\n')
    annotation_file.close()

