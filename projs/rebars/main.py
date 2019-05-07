#!/usr/bin/env python
#coding:utf-8
from pandas import read_csv
from collections import defaultdict
import cv2
import os
from xml.dom.minidom import Document

# 从train_labels.csv文件中读取数据
def load_train_labels(filename=r'data/train_labels.csv'):
    # load rectangles of all training images.
    dataset = read_csv(filename)
    ids = dataset['ID'].get_values()
    rects = dataset['   Detection'].get_values()  # spaces are needed.
    
    img_rects = defaultdict(list)
    for i in range(0, len(ids)):
        img_rects[ids[i]].append(rects[i].split( ))
        
    return img_rects

# 可视化显示,验证数据读取正确性
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
            

def addNode(doc, nodename, parent=None, nodetext=None):
    node = doc.createElement(nodename)
    if nodetext != None:
        text = doc.createTextNode(nodetext)
        node.appendChild(text)

    if parent != None:
        parent.appendChild(node)
    else:
        node.setAttribute("verified", "no")
        doc.appendChild(node)

    return node

# 创建路径
def mkdir(path):
    # 去除首位空格和尾部\符号
    path = path.strip()
    path = path.rstrip("\\")

    # 判断路径是否存在，如果不存在就创建该文件夹
    if not os.path.exists(path):
        os.makedirs(path)

def writeInfor2Xml(filename, folder, path, rects, object_name='chinese', xmlfolder=r'data/train_Annotations/'):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    # 创建dom文档
    doc = Document()

    # 创建根节点
    root_node = addNode(doc, 'annotation')#doc.createElement('annotation')
    # 根节点插入dom树
    #doc.appendChild(root_node)

    # 创建节点<folder>及其文本节点, 并将folder节点插入父节点annotation_list下
    addNode(doc, 'folder', root_node, folder)

    # 创建filename节点
    addNode(doc, 'filename', root_node, filename[:-4])

    # 创建path节点
    addNode(doc, 'path', root_node, path)


    # 创建source节点
    source_node = addNode(doc, 'source', root_node)
    # 创建source节点下的子节点database
    addNode(doc, 'database', source_node, 'Unknown')

    # 创建size节点
    size_node = addNode(doc, 'size', root_node)
    # 创建source节点下的子节点database
    addNode(doc, 'width', size_node, str(img.shape[1]))
    addNode(doc, 'height', size_node, str(img.shape[0]))
    addNode(doc, 'depth', size_node, str(img.shape[2]))

    addNode(doc, 'segmented', root_node, str(0))

    # 开始真是写数据的地方
    for (l,t,r,b) in rects:
        object_node = addNode(doc, 'object', root_node)
        addNode(doc, 'name', object_node, object_name)
        addNode(doc, 'pose', object_node, 'Unspecified')
        addNode(doc, 'truncated', object_node, str(0))
        addNode(doc, 'difficult', object_node, str(0))

        bndbox_node = addNode(doc, 'bndbox', object_node)
        addNode(doc, 'xmin', bndbox_node, str(l))
        addNode(doc, 'ymin', bndbox_node, str(t))
        addNode(doc, 'xmax', bndbox_node, str(r))
        addNode(doc, 'ymax', bndbox_node, str(b))

    # 将dom对象写入本地xml文件
    xmlfile = filename[:-3] + 'xml'
    mkdir(xmlfolder)
    with open(os.path.join(xmlfolder,xmlfile), "w", encoding="utf8") as outfile:
        outfile.write(doc.toprettyxml())


def writexml(img_folder, img_rects, xmlfolder):
    for filename in img_rects:
        path = os.path.join(img_folder, filename)
        folder = img_folder.split('/')
        if len(folder[-1]) == 0:
            folder = folder[-2]
        else:
            folder = folder[-1]

        writeInfor2Xml(filename, folder, path, img_rects[filename], xmlfolder)

# 从csv文件中读取数据,转存为xml格式, 方便后面Yolo3训练网络对数据格式的要求
def covert_csv_to_xml(csvfile=r'data/train_labels.csv', img_folder=r'data/train_dataset', xmlfolder=r'data/train_Annotations/'):
    img_rects = load_train_labels(csvfile)
    writexml(os.path.join(os.getcwd(), img_folder), img_rects, xmlfolder)

# 测试加载并验证数据读取正确性
def load_show():
    img_rects = load_train_labels()
    draw_rects(img_rects)

def main():
    covert_csv_to_xml()


if __name__ == '__main__':
    main()