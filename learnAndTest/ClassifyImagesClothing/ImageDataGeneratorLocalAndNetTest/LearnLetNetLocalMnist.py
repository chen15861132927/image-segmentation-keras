import os
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import glob
import pathlib #读图片路径
import random
from LetNet import LetNet


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

 
def load_and_preprocess_from_path_label(path, label):
    image = tf.io.read_file(path)  # 读取图片
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [28, 28])  # 原始图片大小为(266, 320, 3)，重设为(192, 192)
    image /= 255.0  # 归一化到[0,1]范围

    return image, label
def get_compiled_model(input_shape):
    model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(6, (5,5), padding="same",activation='relu', input_shape=input_shape),
                    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                    tf.keras.layers.ReLU(),  # ReLU激活函数
                    tf.keras.layers.Conv2D(16, (5,5), padding="same",activation='relu', input_shape=input_shape),
                    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                    tf.keras.layers.ReLU(),  # ReLU激活函数
                    tf.keras.layers.Conv2D(120, (5,5), padding="same",activation='relu', input_shape=input_shape),
                    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                    tf.keras.layers.ReLU(),  # ReLU激活函数
                    tf.keras.layers.Flatten(input_shape=input_shape),
                    tf.keras.layers.Dense(84, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])
    model.build(input_shape)
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['acc'])
    model.summary()
    return model

ImageRootPath=os.path.join(Root_Path,"MnistDatagenTrain")
all_images_paths= list(pathlib.Path(ImageRootPath).glob('*/*'))
image_count = len(all_images_paths)

all_images_paths = [str(path) for path in all_images_paths]  # 所有图片路径的列表
random.shuffle(all_images_paths)  # 打散

label_names = sorted(item.name for item in pathlib.Path(ImageRootPath).glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_images_paths]

for image, label in zip(all_images_paths[:5], all_image_labels[:5]):
    print(image, ' --->  ', label)

pathlabds = tf.data.Dataset.from_tensor_slices((all_images_paths, all_image_labels))

image_label_ds  = pathlabds.map(load_and_preprocess_from_path_label)
image_label_ds = image_label_ds.shuffle(buffer_size=64).batch(32)
# Shuffle and slice the dataset.
input_shape=(28, 28, 1)

model = get_compiled_model(input_shape)
# Since the dataset already takes care of batching,
# we don't pass a `batch_size` argument.
model.fit(image_label_ds, epochs=3)

# You can also evaluate or predict on a dataset.
#print("Evaluate")
#result = model.evaluate(image_label_ds)
#dict(zip(model.metrics_names, result))