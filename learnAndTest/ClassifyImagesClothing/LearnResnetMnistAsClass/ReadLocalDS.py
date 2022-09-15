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
    
    return image, label
ImageRootPath=os.path.join(Root_Path,"MnistDatagen")
all_images_paths= list(pathlib.Path(ImageRootPath).glob('*/*'))
all_images_paths = [str(path) for path in all_images_paths]  # 所有图片路径的列表
random.shuffle(all_images_paths)  # 打散

label_names = sorted(item.name for item in pathlib.Path(ImageRootPath).glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_images_paths]

for image, label in zip(all_images_paths[:5], all_image_labels[:5]):
    print(image, ' --->  ', label)

pathlabds = tf.data.Dataset.from_tensor_slices((all_images_paths, all_image_labels))

image_label_ds  = pathlabds.map(load_and_preprocess_from_path_label)


#label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_images_labels,tf.int64)) #读入标签


