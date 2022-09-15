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
    tf_image = tf.reshape(tf_image, [IMAGE_WIDTH, IMAGE_HEIGHT])
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
    label = tf.cast(example["label"], tf.int32)
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
def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(160000)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(80000)
    return dataset


train_dataset = get_dataset("MnisttestTotal.tfrecords")



def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(8):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(str(label_batch[n]))
        plt.axis("off")

image_batch, label_batch = next(iter(train_dataset))
show_batch(image_batch.numpy(), label_batch.numpy())
plt.show()

show_batch(image_batch.numpy(), label_batch.numpy())
plt.show()
image_batch, label_batch = next(iter(train_dataset))
show_batch(image_batch.numpy(), label_batch.numpy())
plt.show()