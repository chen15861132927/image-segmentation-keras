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

#if isTrain=True,labels can't be None
def pics_to_TFRecord(image_label_ds):    
    writer=tf.io.TFRecordWriter("MnisttestTotal.tfrecords")
    totalcount=0
    for train,label in image_label_ds:
        for batchIndex in range(train.shape[0]):
            itemTrain=train[batchIndex]
            itemLabel=label[batchIndex]
            width,height=itemTrain.shape[0],itemTrain.shape[1]
            itemTrain=np.uint8(itemTrain)
            itemTrainbytes=itemTrain.tobytes()
            itemLabel=itemLabel.numpy()
            example=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[itemTrainbytes])),
                        "width":tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                        "height":tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                        "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[itemLabel]))
                    }
                )
            )
            writer.write(record=example.SerializeToString())
            totalcount=totalcount+1
        print("totalcount:",totalcount)
    writer.close()

def load_and_preprocess_from_path_label(path, label):
    image = tf.io.read_file(path)  # 读取图片
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [28, 28])  # 原始图片大小为(266, 320, 3)，重设为(192, 192)
    return image, label

ImageRootPath=os.path.join(Root_Path,"MnistDatagenTest")
all_images_paths= list(pathlib.Path(ImageRootPath).glob('*/*'))
image_count = len(all_images_paths)

all_images_paths = [str(path) for path in all_images_paths]  # 所有图片路径的列表
random.shuffle(all_images_paths)  # 打散

label_names = sorted(item.name for item in pathlib.Path(ImageRootPath).glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_images_paths]

# for image, label in zip(all_images_paths, all_image_labels):
#     print(image, ' --->  ', label)

pathlabds = tf.data.Dataset.from_tensor_slices((all_images_paths, all_image_labels))

image_label_ds = pathlabds.map(partial(load_and_preprocess_from_path_label), num_parallel_calls=tf.data.AUTOTUNE)
image_label_ds = image_label_ds.shuffle(buffer_size=2048).batch(1024)
# Shuffle and slice the dataset.

# You can also evaluate or predict on a dataset.
#print("Evaluate")
#result = model.evaluate(image_label_ds)
#dict(zip(model.metrics_names, result))
#write train record
pics_to_TFRecord(image_label_ds)
