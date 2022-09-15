import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_datasets as tfds


all_size=600
batch_size = 5
X = np.empty([all_size], dtype = int) 
Z = np.empty([all_size], dtype = int) 
# 清空原本列表list1的元素
for i in range(all_size):
    X[i]=i
print(X)
dataset = tf.data.Dataset.from_tensor_slices(X)
# for element in dataset:
#   print(element)
Y = dataset.shuffle(10).batch(batch_size)
for val in Y:
    sort=np.sort(val.numpy())
    print(sort)
    print(val.numpy())
    for valindex in val:
        Z[valindex]=valindex
print(Z)


