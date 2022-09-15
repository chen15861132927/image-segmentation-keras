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

#region load datasets
cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
#class_names = ['0-飞机(airplone)', '1-汽车(automobile)', '2-鸟类(bird)', '3-猫(cat)', '4-鹿(deer)',
#               '5-狗(dog)', '6-蛙类(frog)', '7-马(horse)', '8-船(ship)', '9-卡车(truck)']
class_names = ['0-airplone', '1-automobile', '2-bird', '3-cat', '4-deer',
               '5-dog', '6-frog', '7-horse', '8-ship', '9-truck']
    # 多分类标签生成
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)
# 生成训练数据
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

train_datagan = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.0, 
                rotation_range=15, width_shift_range=0.15, height_shift_range=0.15, fill_mode='wrap')

print(train_images.shape)
print(len(train_labels))
#endregion

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    true_label=np.argmax(true_label)
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],\
                                  100*np.max(predictions_array),\
                                  class_names[np.argmax(true_label)]),\
                                  color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    true_label=np.argmax(true_label)
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[np.argmax(true_label)].set_color('blue')
def create_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(4,4,512)),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

  model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['acc'])

  return model

# 创建一个基本的模型实例
model = create_model()

# 显示模型的结构
model.summary()
EPOCHS = 20
batch_size = 128

#region SaveModel
checkpoint_path = "Simple_CNN_Feature_extraction/cp-{epoch:04d}-{acc:.04f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)

if(latest is not None):
    model.load_weights(latest)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='acc',
    mode='max')

# 加载预训练好的卷积基
conv_base = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
# 用预训练好的卷积基处理训练集提取特征
sample_count = len(train_labels)
train_features_VGG16 = np.zeros(shape=(sample_count, 4, 4, 512))
train_labels_VGG16 = np.zeros(shape=(sample_count, 10))
train_generator_VGG16 = train_datagan.flow(train_images, train_labels, batch_size=batch_size)
i = 0
for inputs_batch, labels_batch in train_generator_VGG16:
    features_batch = conv_base.predict(inputs_batch)
    train_features_VGG16[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels_VGG16[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= sample_count:
        break
# train_features = np.reshape(train_features, (sample_count, 4*4*512))
# 用预训练好的卷积基处理验证集提取特征
sample_count = len(test_labels)
test_generator_VGG16 = train_datagan.flow(test_images, test_labels, batch_size=batch_size)
test_features_VGG16 = np.zeros(shape=(sample_count, 4, 4, 512))
test_labels_VGG16 = np.zeros(shape=(sample_count, 10))
i = 0
for inputs_batch, labels_batch in test_generator_VGG16:
    features_batch = conv_base.predict(inputs_batch)
    test_features_VGG16[i * batch_size : (i + 1) * batch_size] = features_batch
    test_labels_VGG16[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= sample_count:
        break
# test_features = np.reshape(test_features, (sample_count, 4*4*512))

hist = model.fit(train_features_VGG16,train_labels_VGG16,
                  batch_size, 
                  epochs = EPOCHS,
                  validation_data=(test_features_VGG16,test_labels_VGG16), 
                  shuffle=True,
                  callbacks=[model_checkpoint_callback])


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#path = manager.save(checkpoint_number=0)       
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

#region Test Save load weights
# loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# model2 = create_model()
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model2.load_weights(latest)

# # 重新评估模型
# loss,acc = model2.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model2, accuracy: {:5.2f}%".format(100*acc))
#endregion



hist_dict = hist.history
if(len(hist_dict)>0):
  print("train acc:")
  print(hist_dict['acc'])
  print("validation acc:")
  print(hist_dict['val_acc'])
  train_acc = hist.history['acc']
  val_acc = hist.history['val_acc']
  train_loss = hist.history['loss']
  val_loss = hist.history['val_loss']
  # 绘图
  epochs = range(1, len(train_acc)+1)
  plt.figure() # 新建一个图
  plt.plot(epochs, train_acc, 'bo', label = 'Training acc')
  plt.plot(epochs, val_acc, 'r', label = 'Validation acc')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.savefig("accuracy.png")
  plt.figure() # 新建一个图
  plt.plot(epochs, train_loss, 'bo', label = 'Training loss')
  plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.savefig("loss.png")