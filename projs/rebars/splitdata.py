import os
import random

# 创建路径
def mkdir(path):
    # 去除首位空格和尾部\符号
    path = path.strip()
    path = path.rstrip("\\")

    # 判断路径是否存在，如果不存在就创建该文件夹
    if not os.path.exists(path):
        os.makedirs(path)

trainval_percent = 0.1
train_percent = 0.9
xmlfilepath = 'data/train_Annotations'
txtsavepath = 'data/splitset'
mkdir(txtsavepath)
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
print("file's number: {}".format(num))

list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv) # sample(list, k)返回一个长度为k新列表，其中存放list所产生k个随机不重复的元素
train = random.sample(trainval, tr)
print("trainval:", trainval)
print("train:", train)

ftrainval = open(os.path.join(txtsavepath,'trainval.txt'), 'w', encoding='UTF-8')
ftest = open(os.path.join(txtsavepath,'test.txt'), 'w', encoding='UTF-8')
ftrain = open(os.path.join(txtsavepath,'train.txt'), 'w', encoding='UTF-8')
fval = open(os.path.join(txtsavepath,'val.txt'), 'w', encoding='UTF-8')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
