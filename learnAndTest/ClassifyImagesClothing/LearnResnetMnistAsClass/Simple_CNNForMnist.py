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

#region load mnist
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#  0-T恤/上衣,1-裤子,2-套头衫,3-连衣裙,4-外套,5-凉鞋,6-衬衫,7-运动鞋,8-包,9-短靴
class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']
train_images = train_images / 255.0
test_images = test_images / 255.0
print(train_images.shape)
print(len(train_labels))
#endregion

#region show image

""" plt.figure()
plt.imshow(train_images[random.randint(0,len(train_labels))], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show() """
#endregion
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],\
                                  100*np.max(predictions_array),\
                                  class_names[true_label]),\
                                  color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#region SaveModel
checkpoint = tf.train.Checkpoint(model=model)      
checkpoint.restore(tf.train.latest_checkpoint('./SaveModelFormnist128'))
manager = tf.train.CheckpointManager(checkpoint, directory='./SaveModelFormnist128', max_to_keep=3)
#endregion
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10,batch_size=64)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

path = manager.save(checkpoint_number=0)       
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
randList=np.random.randint(len(predictions),size=num_images)
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(randList[i], predictions[randList[i]], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(randList[i], predictions[randList[i]], test_labels)
plt.tight_layout()
plt.show()

