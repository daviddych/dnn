找出二维码中的汉字，使用可参考https://blog.csdn.net/u010429424/article/details/83238988

git clone https://github.com/lijialinneu/keras-yolo3-master.git


```
Usage
```
1. 利用迅雷下载yolo.h5文件 : https://pjreddie.com/media/files/yolov3.weights
2. 格式转换: python convert.py yolov3.cfg yolov3.weights model_data/yolo_weights.h5

model_data/yolo_weights.h5

```
修改配置文件
```
1.修改yolo3.cgf文件:
搜索yolo关键字, 会有三处yolo处, 修改filters,classes 和 random. 具体地:在每一处yolo之前的那个convolutional层都要修改filters的数目，filters=anchors_num * (classes_num + 5),anchors_num为3（一般不变），classes_num为3（根据这个修改就行），修改yolo中classes的数目。注意是每个yolo和yolo前的convolutional层都做相同的修改。random为多尺度训练，1为打开多尺度训练，0为相反。

2.修改keras-yolo3-master\model_data下coco_classes.txt和voc_classes.txt，里面写的是要识别的类型，我写的是chinese。

```
重要模型参数:

```
input_shape:输入图像尺寸；
anchors:检测框尺寸；
num_classes:类别数；
freeze_body: 模式1是全部冻结，模式2是训练最后三层；
weights_path，预训练权重的路径；
logging是TensorBoard的回调，checkpoint是存储权重的回调；
并且，如果电脑配置低的话，需要调小batch_size以及epoch的次数，否则会在unfreeze阶段产生out of memory的错误