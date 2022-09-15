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

classes_num = 10
batch_size = 64
epochs_num = 200

cwd = os.getcwd()
print(cwd)
checkpoint_path="Deep_CNN_Resnet_block/cp-{epoch:04d}-{acc:.04f}.ckpt"
print(os.path.isdir("/content/drive/MyDrive"))
if(cwd.startswith("C:")):
  checkpoint_path = os.path.join(cwd,checkpoint_path)
elif(cwd.startswith("/content")):
  if(os.path.isdir("/content/drive/MyDrive") is False):
    from google.colab import drive
    drive.mount('/content/drive')
  checkpoint_path = os.path.join(cwd,"/content/drive/MyDrive/"+checkpoint_path)
print(checkpoint_path) 
#endregion
#region load datasets

cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
#class_names = ['0-飞机(airplone)', '1-汽车(automobile)', '2-鸟类(bird)', '3-猫(cat)', '4-鹿(deer)',
#               '5-狗(dog)', '6-蛙类(frog)', '7-马(horse)', '8-船(ship)', '9-卡车(truck)']
class_names = ['0-airplone', '1-automobile', '2-bird', '3-cat', '4-deer',
               '5-dog', '6-frog', '7-horse', '8-ship', '9-truck']
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = train_images / 255.0
test_images = test_images / 255.0


# 多分类标签生成
train_labels =tf. keras.utils.to_categorical(train_labels, classes_num)
test_labels = tf.keras.utils.to_categorical(test_labels, classes_num)
print(train_images.shape)
print(len(train_labels))

#endregion

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    true_label=true_label.argmax()
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
    true_label=true_label.argmax()
    plt.grid(False)
    plt.xticks(range(classes_num))
    plt.yticks([])
    thisplot = plt.bar(range(classes_num), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def resnet_block(inputs, num_filters=16, kernel_size=3,strides=1, activation='relu'):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
           kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if (activation):
        x = tf.keras.layers.Activation('relu')(x)
    return x

# 建一个20层的ResNet网络 
def create_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)# Input层，用来当做占位使用
    
    #第一层
    x = resnet_block(inputs)
    print('layer1,xshape:',x.shape)
    # 第2~7层
    for i in range(6):
        a = resnet_block(inputs = x)
        b = resnet_block(inputs=a,activation=None)
        x = tf.keras.layers.add([x,b])
        x = tf.keras.layers.Activation('relu')(x)
    # out：32*32*16
    # 第8~13层
    for i in range(6):
        if i == 0:
            a = resnet_block(inputs = x,strides=2,num_filters=32)
        else:
            a = resnet_block(inputs = x,num_filters=32)
        b = resnet_block(inputs=a,activation=None,num_filters=32)
        if i==0:
            x = tf.keras.layers.Conv2D(32,kernel_size=3,strides=2,padding='same',
                       kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.add([x,b])
        x = tf.keras.layers.Activation('relu')(x)
    # out:16*16*32
    # 第14~19层
    for i in range(6):
        if i ==0 :
            a = resnet_block(inputs = x,strides=2,num_filters=64)
        else:
            a = resnet_block(inputs = x,num_filters=64)

        b = resnet_block(inputs=a,activation=None,num_filters=64)
        if i == 0:
            x = tf.keras.layers.Conv2D(64,kernel_size=3,strides=2,padding='same',
                       kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.add([x,b])# 相加操作，要求x、b shape完全一致
        x = tf.keras.layers.Activation('relu')(x)
    # out:8*8*64
    # 第20层   
    x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
    # out:4*4*64
    y = tf.keras.layers.Flatten()(x)
    # out:1024
    outputs = tf.keras.layers.Dense(classes_num,activation='softmax',
                    kernel_initializer='he_normal')(y)
    
    #初始化模型
    #之前的操作只是将多个神经网络层进行了相连，通过下面这一句的初始化操作，才算真正完成了一个模型的结构初始化
    model = tf.keras.models.Model(inputs=inputs,outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])

    return model

# 动态变化学习率
def lr_sch(epoch):
    #200 total
    if epoch <50:
        return 1e-3
    if 50<=epoch<100:
        return 1e-4
    if epoch>=100:
        return 1e-5
# 创建一个基本的模型实例
model = create_model((32,32,3))

# 显示模型的结构
model.summary()

#region SaveModel
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
if(latest is not None):
    model.load_weights(latest)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_acc', verbose=1,save_best_only=True)
lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_sch)
lr_reducer_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, mode='max', min_lr=1e-3)
callbacks = [model_checkpoint_callback, lr_scheduler_callback, lr_reducer_callback]

hist = model.fit(train_images, train_labels, 
                epochs=epochs_num, 
                batch_size=batch_size,
                validation_data=(test_images,test_labels),
                callbacks=callbacks)

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



#model.save('cifar10_model.hdf5') 

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