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

def decode_fn(record_bytes):
  return tf.io.parse_single_example(
      # Data
      record_bytes,

      # Schema
      {
        "image": tf.io.FixedLenFeature([], dtype=tf.string),
        "width": tf.io.FixedLenFeature([], dtype=tf.float32),
        "height": tf.io.FixedLenFeature([], dtype=tf.float32),
        "label": tf.io.FixedLenFeature([], dtype=tf.float32)
      }
  )
def decode_image(itemTrainbytes):
    tf_image = tf.io.decode_raw(itemTrainbytes, tf.uint8)
    tf_image = tf.reshape(tf_image, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])
    #tf_image = tf_image.astype('float32') / 255.0
    #tf_image = np.expand_dims(tf_image, -1)
    return tf_image
def read_tfrecord(example):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
        # if labeled
        # else {"image": tf.io.FixedLenFeature([], tf.string),}
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    # if labeled:
    label = example["label"]
    #label=label.numpy()
    return image, label
    # return image
def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord), num_parallel_calls=tf.data.AUTOTUNE
    )
    #dataset = dataset.map(read_tfrecord)
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset
def get_dataset(filenames,shuffle,totalcount):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(totalcount)
    return dataset
batch_size=128

train_dataset = get_dataset("MnisttrainTotal.tfrecords",900000,512000)
test_dataset = get_dataset("MnisttestTotal.tfrecords",120000,80000)

x_train, y_train = next(iter(train_dataset))
x_test, y_test = next(iter(test_dataset))
ResnetTrain= LetNet(subtract_pixel_mean=False)
input_shape=(28, 28, 1)
epochs = 50

# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)
# strategy = tf.distribute.experimental.TPUStrategy(tpu)

# with strategy.scope():
ResnetTrain.compile(input_shape)

ResnetTrain.Root_Path=Root_Path
ResnetTrain.train(x_train=np.copy(x_train), y_train=y_train.numpy(),x_test=np.copy(x_test),
 y_test=y_test.numpy(),epochs=epochs,batch_size=batch_size,Nodata_augmentation=True,
 modelFile='/MnistLetNet38v2_modelcp-0128-0.9953.h5')

# 评估训练模型
#ResnetTrain.evaluateAndSaveFail(x_test, y_test,batch_size=batch_size,evaluateName='test')
