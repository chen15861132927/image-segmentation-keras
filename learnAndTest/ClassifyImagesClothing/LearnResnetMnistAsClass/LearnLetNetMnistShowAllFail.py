import os
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
from LetNet import LetNet

cwd = os.getcwd()
print(cwd)
Root_Path=sys.path[0]
print(os.path.isdir("/content/drive/MyDrive"))
if(cwd.startswith("/content")):
  if(os.path.isdir("/content/drive/MyDrive") is False):
    from google.colab import drive
    drive.mount('/content/drive')
  Root_Path = os.path.join(cwd,"/content/drive/MyDrive/")
print(Root_Path) 

# 训练参数
batch_size = 128  # 原论文按照 batch_size=128 训练所有的网络
epochs = 0
num_classes = 10

# 载入 mnist 数据。
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 输入图像维度。
input_shape = x_train.shape[1:]
if(len(input_shape)==2):
  input_shape=input_shape + (1,)
if(len(input_shape)==3):
  input_shape=(batch_size,)+input_shape 

input_shape=(28, 28, 1)
# 数据标准化。
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print('x_train shape:', x_train.shape,x_train.shape[0], 'train samples')
print('y_train shape:', y_train.shape,x_test.shape[0], 'test samples')

# 将类向量转换为二进制类矩阵。
#y_train = tf.keras.utils.to_categorical(y_train, num_classes)
#y_test = tf.keras.utils.to_categorical(y_test, num_classes)

ResnetTrain= LetNet(subtract_pixel_mean=True)

# tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)
# strategy = tf.distribute.experimental.TPUStrategy(tpu)

# with strategy.scope():
ResnetTrain.compile(input_shape)

ResnetTrain.Root_Path=Root_Path
ResnetTrain.train(x_train=np.copy(x_train), y_train=y_train,x_test=np.copy(x_test),
 y_test=y_test,epochs=epochs,batch_size=batch_size,Nodata_augmentation=True,
 modelFile='/LetNetMnist_model-epoch0002-acc0.963-val_acc0.9862.h5')

# 评估训练模型
ResnetTrain.evaluateAndSaveFail(x_test, y_test,batch_size=batch_size,evaluateName='test')

#ResnetTrain.evaluateAndSaveFail(x_train, y_train,batch_size,'train')

