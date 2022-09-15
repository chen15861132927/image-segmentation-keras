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

list=[]
for index in range(100):
    list.append(index)

list=np.array(list)

dataset = tf.data.Dataset.from_tensor_slices(list)
dataset = dataset.shuffle(buffer_size=40).batch(20)

# iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
# one_element = iterator.get_next()
# for i in range(100):
#     print(one_element)
for one_element in dataset.as_numpy_iterator():
    print(one_element)