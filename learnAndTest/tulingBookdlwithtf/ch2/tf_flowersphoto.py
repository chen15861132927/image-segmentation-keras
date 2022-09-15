import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_datasets as tfds


num_epoch = 5
batch_size = 50
learning_rate = 0.001
IMAGE_WIDTH=IMAGE_HEIGHT=224
dataset = tfds.load("tf_flowers", split="train", as_supervised=True)
# dataset = dataset.map(lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label)).shuffle(1024).batch(batch_size)
# dataset = dataset.map(lambda img, label: (img , label)).shuffle(1024).batch(batch_size)
batch_index=0
for images, labels in dataset:
    imagesnumpy=images.numpy()
    labelsnumpy=labels.numpy()
    #for xtindex in range(0,imagesnumpy.shape[0]):
    #img = imgarray.reshape(IMAGE_WIDTH, IMAGE_HEIGHT,3)  # 把图像的形状变为原来的尺寸
    pil_img = Image.fromarray(np.uint8(imagesnumpy), mode='RGB')
    # plt.imshow(pil_img) # 显示图片
    # plt.axis('off') # 不显示坐标轴
    # plt.show()
    path=sys.path[0]+"\\tf_flowers\\"+str(labelsnumpy)
    if(os.path.exists(path)==False):
        os.makedirs(path) 
    fileName=path+"\\"+str(labelsnumpy)+"_"+str(batch_index)+".jpg"
    pil_img.save(fileName, "JPEG")
    print(fileName)
    batch_index=batch_index+1