import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import glob
import pathlib #读图片路径
import random
import numpy as np
from LetNet import LetNet
from functools import partial

cwd = os.getcwd()
print(cwd)
Root_Path=sys.path[0]
googlePath="/content/drive/MyDrive/"
print(os.path.isdir(googlePath))
if(cwd.startswith("/content")):
  if(os.path.isdir(googlePath) is False):
    from google.colab import drive
    drive.mount('/content/drive')
  Root_Path = os.path.join(cwd,googlePath)
print(Root_Path) 
IMAGE_WIDTH = IMAGE_HEIGHT = 28
#path(path of folder)
path=r'C:/Dev/GithubProject/dlwithtf/ClassifyImagesClothing/LearnResnetMnistAsClass/MnistDatagen/0/0_1_1.jpg'
image = tf.io.read_file(path)  # 读取图片
image = tf.image.decode_jpeg(image, channels=1)
image = tf.image.resize(image, [28, 28])  # 原始图片大小为(266, 320, 3)，重设为(192, 192)

itemTrain=tf.reshape(image,[IMAGE_WIDTH, IMAGE_HEIGHT])
width,height=itemTrain.shape[0],itemTrain.shape[1]
itemTrain=np.uint8(itemTrain)
itemTrainbytes=itemTrain.tobytes()

#name=pathlib.Path(path).glob('*/') 
label=pathlib.Path(path).parent.name
# itemLabel=0
# itemLabel=itemLabel.numpy()
tf_image = tf.io.decode_raw(itemTrainbytes, tf.uint8)
tf_image = tf.reshape(tf_image, [IMAGE_WIDTH, IMAGE_HEIGHT,1])
#tf_image = np.expand_dims(tf_image, -1)
#tf_image = tf_image.astype('float32') / 255.0


# Loadimage = tf.image.decode_jpeg(tf_image, channels=1)
Loadimage = tf.cast(tf_image, tf.float32)
Loadimage= tf.divide(Loadimage,255.0)
# Loadimage = tf.reshape(image, [28, 28])

plt.imshow(Loadimage)
plt.title("BENIGN")
plt.axis("off")
plt.show()
