#region import
import os
# 导入 random(随机数) 模块
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# TensorFlow and tf.keras
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

#endregion

#region load fashion_mnist
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#  0-T恤/上衣,1-裤子,2-套头衫,3-连衣裙,4-外套,5-凉鞋,6-衬衫,7-运动鞋,8-包,9-短靴
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0
print(train_images.shape)
print(len(train_labels))
#endregion

def create_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'])

  return model

# 创建一个基本的模型实例
model = create_model()

# 显示模型的结构
model.summary()

checkpoint_path = "SimpleCNNForfashion_mnist/cp-{epoch:04d}-{acc:.04f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)

if(latest is not None):
    model.load_weights(latest)

EPOCHS = 2
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True)

# 使用新的回调训练模型
model.fit(train_images, 
          train_labels,  
          epochs=EPOCHS,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # 通过回调训练
          
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
