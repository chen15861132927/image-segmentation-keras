import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_datasets as tfds

X=tf.constant([[1, 2],[ 3, 4]])
b=tf.constant([1,1,3])

Xnumpy=X.numpy()
Xdim = tf.expand_dims(X, -1)
img3=tf.tile(Xdim,b)
# image = tf.zeros([10,10,3])
# disms=tf.expand_dims(image, image.shape.ndims)
# disms.shape.as_list()
imagesnumpy=img3.numpy()

print(img3.shape)