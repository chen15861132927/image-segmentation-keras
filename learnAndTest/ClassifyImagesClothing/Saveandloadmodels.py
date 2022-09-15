import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

#region load fashion_mnist
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#  0-T恤/上衣,1-裤子,2-套头衫,3-连衣裙,4-外套,5-凉鞋,6-衬衫,7-运动鞋,8-包,9-短靴
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0


# 定义一个简单的序列模型
def create_model():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

# 创建一个基本的模型实例
model = create_model()

# 显示模型的结构
model.summary()

checkpoint_path = "training_6/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)

if(latest is not None):
    model.load_weights(latest)
# 创建一个保存模型权重的回调
# 创建一个回调，每 5 个 epochs 保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True)
model.save_weights(checkpoint_path.format(epoch=0))

# 使用新的回调训练模型
model.fit(train_images, 
          train_labels,  
          epochs=3,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # 通过回调训练

# 这可能会生成与保存优化程序状态相关的警告。
# 这些警告（以及整个笔记本中的类似警告）
# 是防止过时使用，可以忽略。

loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


model2 = create_model()
latest = tf.train.latest_checkpoint(checkpoint_dir)

model2.load_weights(latest)

# 重新评估模型
loss,acc = model2.evaluate(test_images,  test_labels, verbose=2)
print("Restored model2, accuracy: {:5.2f}%".format(100*acc))


