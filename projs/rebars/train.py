"""
Retrain the YOLO model for your own dataset.

Visualization training process
1. cd rebars
2. Press "Ctrl+Alt+T" and type "tensorboard --logdir logs" in terminal.
3. copy the link from terminal, and paste it into chrome, eg: http://dyc:6006
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
import os
import re

# 创建路径
def mkdir(path):
    # 去除首位空格和尾部\符号
    path = path.strip()
    path = path.rstrip("\\")

    # 判断路径是否存在，如果不存在就创建该文件夹
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def remove_files(log_dir, log_ext='dyc'):
    ''' 删除指定文件夹下,特定后缀的文件 '''
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            if re.match(r'.*\.({})'.format(log_ext), f, flags=re.I):
                os.remove(os.path.join(log_dir, f))

def _main():
    annotation_path = 'train.txt'
    log_dir = mkdir('logs')
    classes_path = 'model/voc_classes.txt'
    anchors_path = 'model/yolo_anchors.txt'
    #pre_weights_path = 'model/yolo_weights.h5'
    pre_weights_path = 'logs/trained_weights_2.h5'
    save_weights_path = 'trained_weights.h5'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    input_shape = (416,416) # multiple of 32, hw
    remove_files('logs', 'dyc')

    # 加载yolo_weights.h5文件
    model = create_model(input_shape, anchors, len(class_names), load_pretrained=True, weights_path = pre_weights_path)

    train(model, annotation_path, input_shape, anchors, len(class_names), log_dir=log_dir, save_weights_path=save_weights_path)

def train(model, annotation_path, input_shape, anchors, num_classes, log_dir=r'logs', save_weights_path='trained_weights.h5'):
    model.compile(optimizer='adam', loss={
        'yolo_loss': lambda y_true, y_pred: y_pred})
    logging = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)
    checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    batch_size = 8
    val_split = 0.1
    with open(annotation_path, encoding='UTF-8') as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))


    try :
        #########2、修改epochs为30 ###########
        model.fit_generator(data_generator_wrap(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch = max(1, num_train // batch_size),
            validation_data = data_generator_wrap(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps = max(1, num_val // batch_size), epochs = 200, initial_epoch = 0, verbose=1,
                            callbacks=[logging])
    except :
        print("error")
    finally:
        model.save_weights(os.path.join(log_dir, save_weights_path[:-3] + '_except.h5')) # 'trained_weights_except.h5'

    model.save_weights(os.path.join(log_dir, save_weights_path))

# 从配置文件中解析类标签名
def get_classes(classes_path):
    with open(classes_path, encoding='UTF-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    with open(anchors_path, encoding='UTF-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=False, weights_path='model/yolo_weights.h5'):
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body:
            # Do not freeze 3 output layers.
            num = len(model_body.layers)-7
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, max_boxes=300):
    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape, max_boxes=max_boxes, random=True)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()