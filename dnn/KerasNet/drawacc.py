import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt
from livelossplot.keras import PlotLossesCallback  # pip install livelossplot


class History(keras.callbacks.Callback):
    """ 定义History类保存训练过程中的loss和acc信息 """

    def __init__(self):
        """ 创建字典盛放loss 和 acc """
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 重写父类方法
    def on_batch_end(self, batch, logs={}):
        # 每个batch完成后向字典中追加loss和acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    # 重写父类方法
    def on_train_end(self, logs=None):
        self.update_frame("epoch")
        plt.savefig("acc-loss-" + 'epoch' + '.jpg')

        self.update_frame("batch")
        plt.savefig("acc-loss-" + 'batch' + '.jpg')

    # 重写父类方法
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

        self.update_frame("epoch")
        plt.pause(0.005)

    # 更新绘图内容
    def update_frame(self, loss_type):
        plt.cla
        x = range(len(self.losses[loss_type]))
        # acc
        plt.plot(x, self.accuracy[loss_type], 'r', label='train acc')  # plt.plot(x,y)，这个将数据画成曲线
        #  loss
        plt.plot(x, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(x, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(x, self.val_loss[loss_type], 'k', label='val loss')

    # 绘图并显示
    def plot_show(self, loss_type):
        self.update_frame(loss_type)
        plt.legend(loc="upper right")  # 设置图例显示位置
        plt.grid(True)  # 设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')  # 给x，y轴加注释
        plt.title("acc-loss/" + loss_type)

        plt.show()


batch_size = 128
nb_classes = 10
nb_epoch = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


plot_losses = PlotLossesCallback()


# 创建一个实例LossHistory
#history = History()
model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=0,
          validation_data=(X_test, Y_test),
          callbacks=[plot_losses])   # 回调函数将数据传给history, 此处必须是数组

# 模型评估
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# 绘制acc-loss曲线


# history.plot_show('batch')
# history.plot_show('epoch')
